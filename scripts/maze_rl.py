from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
MAZE = ROOT / "data" / "maze" / "maze.jpg"
if not MAZE.exists():
    MAZE = ROOT / "附件" / "maze.jpg"
OUT = ROOT / "outputs" / "maze"
OUT.mkdir(parents=True, exist_ok=True)


def load_grid():
    img = cv2.imread(str(MAZE), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = gray > 105
    ys, xs = np.where(bright)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1]

    yellow = (
        (crop[:, :, 0] > 120)
        & (crop[:, :, 1] > 100)
        & (crop[:, :, 0] > 1.35 * crop[:, :, 2])
        & (crop[:, :, 1] > 1.20 * crop[:, :, 2])
    )
    sy, sx = np.argwhere(yellow).mean(axis=0).astype(int)

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    free = gray_crop < 85
    free[yellow] = True

    start = nearest_free(free, (sy, sx))
    return crop, free, start


def nearest_free(free, seed):
    q = deque([seed])
    seen = {seed}
    h, w = free.shape
    while q:
        y, x = q.popleft()
        if 0 <= y < h and 0 <= x < w and free[y, x]:
            return y, x
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen:
                seen.add((ny, nx))
                q.append((ny, nx))
    raise RuntimeError("No free cell found.")


def farthest_reachable(free, start):
    q = deque([start])
    dist = {start: 0}
    farthest = start
    while q:
        y, x = q.popleft()
        if dist[(y, x)] > dist[farthest]:
            farthest = (y, x)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (y + dy, x + dx)
            if (
                0 <= nb[0] < free.shape[0]
                and 0 <= nb[1] < free.shape[1]
                and free[nb]
                and nb not in dist
            ):
                dist[nb] = dist[(y, x)] + 1
                q.append(nb)
    return farthest


def reachable_distances(free, start):
    q = deque([start])
    dist = {start: 0}
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (y + dy, x + dx)
            if (
                0 <= nb[0] < free.shape[0]
                and 0 <= nb[1] < free.shape[1]
                and free[nb]
                and nb not in dist
            ):
                dist[nb] = dist[(y, x)] + 1
                q.append(nb)
    return dist


def default_goal_cases(free, start):
    dist = reachable_distances(free, start)
    ordered = sorted(dist.items(), key=lambda item: item[1])
    fractions = [0.10, 0.50, 1.00]
    cases = []
    for label, frac in zip(("reachable-near", "reachable-mid", "reachable-far"), fractions):
        y, x = ordered[int((len(ordered) - 1) * frac)][0]
        cases.append((label, (x, y)))

    ys, xs = np.where(free)
    for y, x in zip(ys, xs):
        y, x = int(y), int(x)
        if (y, x) not in dist and 30 < x < free.shape[1] - 30 and 30 < y < free.shape[0] - 30:
            cases.append(("unreachable-free", (x, y)))
            break

    wall = None
    for x, y in ((100, 100), (50, 50), (250, 120)):
        if 0 <= x < free.shape[1] and 0 <= y < free.shape[0] and not free[y, x]:
            wall = (x, y)
            break
    if wall is None:
        for y in range(30, free.shape[0] - 30):
            for x in range(30, free.shape[1] - 30):
                if not free[y, x]:
                    wall = (x, y)
                    break
            if wall is not None:
                break
    cases.append(("blocked-wall", wall))
    return cases


def bfs(free, start, goal):
    q = deque([start])
    prev = {start: None}
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            break
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (y + dy, x + dx)
            if (
                0 <= nb[0] < free.shape[0]
                and 0 <= nb[1] < free.shape[1]
                and free[nb]
                and nb not in prev
            ):
                prev[nb] = (y, x)
                q.append(nb)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def value_iteration(free, goal, gamma=0.995):
    h, w = free.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    dist[goal] = 0.0
    queue = deque([goal])
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        y, x = queue.popleft()
        for dy, dx in actions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and not np.isfinite(dist[ny, nx]):
                dist[ny, nx] = dist[y, x] + 1.0
                queue.append((ny, nx))

    v = np.full((h, w), -1.0e6, dtype=np.float32)
    reachable = np.isfinite(dist)
    v[reachable] = -dist[reachable]
    v[goal] = 100.0
    q = np.full((h, w, 4), -1.0e6, dtype=np.float32)
    ys, xs = np.where(free)
    for y, x in zip(ys, xs):
        for a, (dy, dx) in enumerate(actions):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and free[ny, nx]:
                reward = 100.0 if (ny, nx) == goal else -1.0
                q[y, x, a] = reward + gamma * v[ny, nx]
            else:
                q[y, x, a] = -6.0 + gamma * v[y, x]
    return q


def greedy_path(q, free, start, goal, max_steps=10000):
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    cur = start
    path = [cur]
    seen = {cur}
    for _ in range(max_steps):
        if cur == goal:
            return path
        y, x = cur
        order = np.argsort(q[y, x])[::-1]
        moved = False
        for a in order:
            dy, dx = actions[int(a)]
            nb = (y + dy, x + dx)
            if 0 <= nb[0] < free.shape[0] and 0 <= nb[1] < free.shape[1] and free[nb]:
                cur = nb
                path.append(cur)
                moved = True
                break
        if not moved or cur in seen:
            return None
        seen.add(cur)
    return None


def solve_goal(free, start, goal_xy):
    x, y = goal_xy
    h, w = free.shape
    result = {
        "goal_xy": goal_xy,
        "status": "",
        "reachable": False,
        "path": None,
        "bfs_length": None,
        "bellman_length": None,
        "used_fallback": None,
    }
    if not (0 <= x < w and 0 <= y < h):
        result["status"] = "out_of_bounds"
        return result
    goal = (y, x)
    if not free[goal]:
        result["status"] = "blocked_or_wall"
        return result

    path_bfs = bfs(free, start, goal)
    if path_bfs is None:
        result["status"] = "unreachable"
        return result

    q = value_iteration(free, goal)
    path_rl = greedy_path(q, free, start, goal, max_steps=max(10000, free.size))
    used_fallback = False
    if path_rl is None or path_rl[-1] != goal or len(path_rl) != len(path_bfs):
        path_rl = path_bfs
        used_fallback = True

    result.update(
        {
            "status": "reachable",
            "reachable": True,
            "path": path_rl,
            "bfs_length": len(path_bfs) - 1,
            "bellman_length": len(path_rl) - 1,
            "used_fallback": used_fallback,
        }
    )
    return result


def draw_path(crop, path, start, goal, path_bfs, path_name="maze_path.png", color=(230, 50, 40)):
    img = Image.fromarray(crop).convert("RGB")
    draw = ImageDraw.Draw(img)
    if path_bfs:
        pts = [(x, y) for y, x in path_bfs]
        draw.line(pts, fill=(30, 144, 255), width=2)
    if path:
        pts = [(x, y) for y, x in path]
        draw.line(pts, fill=color, width=3)
    sy, sx = start
    gy, gx = goal
    draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), fill=(255, 220, 0), outline=(0, 0, 0))
    draw.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), fill=(0, 200, 90), outline=(0, 0, 0))
    img.save(OUT / path_name)


def draw_invalid_goals(crop, start, results):
    img = Image.fromarray(crop).convert("RGB")
    draw = ImageDraw.Draw(img)
    sy, sx = start
    draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill=(255, 220, 0), outline=(0, 0, 0))
    colors = {"unreachable": (148, 103, 189), "blocked_or_wall": (255, 127, 14)}
    for result in results:
        if result["reachable"]:
            continue
        x, y = result["goal_xy"]
        if 0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]:
            color = colors.get(result["status"], (214, 39, 40))
            draw.line((x - 8, y - 8, x + 8, y + 8), fill=color, width=4)
            draw.line((x - 8, y + 8, x + 8, y - 8), fill=color, width=4)
    img.save(OUT / "maze_invalid_goals.png")


def draw_multi_paths(crop, start, results):
    img = Image.fromarray(crop).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = [
        (230, 50, 40),
        (31, 119, 180),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
    ]
    for idx, result in enumerate(results):
        x, y = result["goal_xy"]
        color = colors[idx % len(colors)]
        if result["path"]:
            pts = [(px, py) for py, px in result["path"]]
            draw.line(pts, fill=color, width=2)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(0, 0, 0))
        elif 0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]:
            draw.line((x - 6, y - 6, x + 6, y + 6), fill=color, width=3)
            draw.line((x - 6, y + 6, x + 6, y - 6), fill=color, width=3)
    sy, sx = start
    draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill=(255, 220, 0), outline=(0, 0, 0))
    img.save(OUT / "maze_paths_multi_goal.png")


def main():
    crop, free, start = load_grid()
    cases = default_goal_cases(free, start)
    results = []
    for label, goal_xy in cases:
        result = solve_goal(free, start, goal_xy)
        result["label"] = label
        results.append(result)

    reachable_results = [r for r in results if r["reachable"]]
    colors = [(230, 50, 40), (245, 140, 30), (125, 85, 200)]
    if reachable_results:
        for idx, result in enumerate(reachable_results):
            goal_yx = (result["goal_xy"][1], result["goal_xy"][0])
            safe_label = result["label"].replace("-", "_")
            draw_path(
                crop,
                result["path"],
                start,
                goal_yx,
                bfs(free, start, goal_yx),
                path_name=f"maze_path_{safe_label}.png",
                color=colors[idx % len(colors)],
            )
        first = reachable_results[0]
        first_goal_yx = (first["goal_xy"][1], first["goal_xy"][0])
        draw_path(crop, first["path"], start, first_goal_yx, bfs(free, start, first_goal_yx))
    draw_multi_paths(crop, start, results)
    draw_invalid_goals(crop, start, results)

    lines = [
        "Maze reinforcement learning experiment",
        f"Grid size: {free.shape[1]} x {free.shape[0]} pixels",
        f"Free cells: {int(free.sum())}",
        f"Start (x,y): ({start[1]}, {start[0]})",
        "",
        "label,goal_x,goal_y,status,bfs_length,bellman_length,used_fallback",
    ]
    for result in results:
        x, y = result["goal_xy"]
        lines.append(
            f"{result['label']},{x},{y},{result['status']},"
            f"{result['bfs_length'] if result['bfs_length'] is not None else 'NA'},"
            f"{result['bellman_length'] if result['bellman_length'] is not None else 'NA'},"
            f"{result['used_fallback'] if result['used_fallback'] is not None else 'NA'}"
        )
    (OUT / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
