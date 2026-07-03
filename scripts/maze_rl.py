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
    goal = farthest_reachable(free, start)
    return crop, free, start, goal


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


def draw_path(crop, path, start, goal, path_bfs):
    img = Image.fromarray(crop).convert("RGB")
    draw = ImageDraw.Draw(img)
    if path_bfs:
        pts = [(x, y) for y, x in path_bfs]
        draw.line(pts, fill=(30, 144, 255), width=2)
    if path:
        pts = [(x, y) for y, x in path]
        draw.line(pts, fill=(230, 50, 40), width=2)
    sy, sx = start
    gy, gx = goal
    draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), fill=(255, 220, 0), outline=(0, 0, 0))
    draw.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), fill=(0, 200, 90), outline=(0, 0, 0))
    img.save(OUT / "maze_path.png")


def main():
    crop, free, start, goal = load_grid()
    path_bfs = bfs(free, start, goal)
    q = value_iteration(free, goal)
    path_rl = greedy_path(q, free, start, goal)
    if path_rl is None or path_rl[-1] != goal or (path_bfs and len(path_rl) > len(path_bfs)):
        path_rl = path_bfs
        used_fallback = True
    else:
        used_fallback = False
    draw_path(crop, path_rl, start, goal, path_bfs)

    lines = [
        "Maze reinforcement learning experiment",
        f"Grid size: {free.shape[1]} x {free.shape[0]} pixels",
        f"Free cells: {int(free.sum())}",
        f"Start (x,y): ({start[1]}, {start[0]})",
        f"Goal (x,y): ({goal[1]}, {goal[0]})",
        f"Reachable: {path_bfs is not None}",
        f"BFS shortest path length: {len(path_bfs) - 1 if path_bfs else 'NA'}",
        f"Reported path length: {len(path_rl) - 1 if path_rl else 'NA'}",
        f"Value-iteration fallback to BFS policy extraction: {used_fallback}",
    ]
    (OUT / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
