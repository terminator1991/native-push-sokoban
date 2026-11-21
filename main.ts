// Room-by-room push-optimal Sokoban solver.
//
// The levels this contest draws are built from independent rooms off a one-wide hallway: a box can
// never leave the room it starts in (pushed through the door it would sit in the hallway, from where
// no goal is reachable), so a level's push optimum is the sum of its rooms' optima. This solver
// finds the rooms, then solves each one push-optimally with an A* search over pushes:
//
//   - a state is the sorted box squares plus the player's reachable region (named by its lowest
//     square), so player walks between pushes are never part of the state;
//   - the estimate is a minimum-cost matching of boxes to goals over true push distances (the
//     cheapest pushes that carry one box to one goal, other boxes ignored), which never overstates
//     the pushes still needed, so the first solution found is push-optimal;
//   - pushes that land a box on a dead square (one no push can bring to any goal) or freeze a 2x2
//     block of walls and boxes around an off-goal box are pruned: neither can be part of a solution.
//
// Everything is budgeted by work (states expanded), never by the clock, and every tie is broken by
// square order, so two runs write byte-identical answers. A room the search somehow cannot finish
// falls back to the whole level, and a level that still resists is answered with the rooms that did
// solve, so every instance is always answered in format.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in a fixed order: up, down, left, right. A step letter is lower case, a push upper.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];

const ROOM_BUDGET = 600_000; // states one room's search may expand before it gives up
const LEVEL_BUDGET = 1_500_000; // states for the whole-level fallback search

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

/** A level in XSB text: width, height, walls (the outer frame is always wall), goals, boxes, player. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * For each goal, the fewest pushes that carry a lone box from any square to that goal, walls the
 * only obstacle: -1 where the goal cannot be reached. Worked backwards from the goal: a box now on
 * x could have been pushed there from x - d, with the player standing on x - 2d.
 */
function pushDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array[] {
  const size = wall.length;
  return goals.map((g) => {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const y = x - d;
        if (y < 0 || y >= size || dist[y] !== -1) continue;
        if (wall[y] || wall[y - d]) continue;
        dist[y] = dist[x] + 1;
        queue.push(y);
      }
    }
    return dist;
  });
}

/** All orderings of 0..n-1, in lexicographic order, for the box-to-goal matching estimate. */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const build = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      build();
      cur.pop();
      used[i] = false;
    }
  };
  build();
  return out;
}

/** A binary heap of state ids, ordered by (f, then deeper g, then insertion order): deterministic. */
class OpenList {
  private f: number[] = [];
  private g: number[] = [];
  private seq: number[] = [];
  private id: number[] = [];
  private counter = 0;

  get size(): number {
    return this.id.length;
  }

  private before(a: number, b: number): boolean {
    if (this.f[a] !== this.f[b]) return this.f[a] < this.f[b];
    if (this.g[a] !== this.g[b]) return this.g[a] > this.g[b];
    return this.seq[a] < this.seq[b];
  }

  private swap(a: number, b: number): void {
    for (const arr of [this.f, this.g, this.seq, this.id] as number[][]) {
      const t = arr[a];
      arr[a] = arr[b];
      arr[b] = t;
    }
  }

  push(f: number, g: number, id: number): void {
    this.f.push(f);
    this.g.push(g);
    this.seq.push(this.counter++);
    this.id.push(id);
    let i = this.id.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.before(i, p)) break;
      this.swap(i, p);
      i = p;
    }
  }

  pop(): number {
    const top = this.id[0];
    const last = this.id.length - 1;
    this.swap(0, last);
    this.f.pop();
    this.g.pop();
    this.seq.pop();
    this.id.pop();
    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < this.id.length && this.before(l, m)) m = l;
      if (r < this.id.length && this.before(r, m)) m = r;
      if (m === i) break;
      this.swap(i, m);
      i = m;
    }
    return top;
  }
}


/**
 * The push-optimal solution of one (sub-)level: the given boxes to the given goals on the given
 * walls, the player starting on `player`. Returns the full move string (walks lower case, pushes
 * upper), or null when the search gives up within its expansion caps or no solution exists. `cap`
 * bounds this call's expansions; `shared` is the set-wide expansion budget both draw from, so the
 * whole run's work is bounded by counts, never by the clock.
 */
function solvePushes(
  w: number,
  wall: Uint8Array,
  goals: number[],
  startBoxes: number[],
  player: number,
  cap: number,
  shared: { left: number },
): string | null {
  const size = wall.length;
  const n = startBoxes.length;
  const goalSet = new Set(goals);
  const tables = pushDistances(w, wall, goals);
  const live = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    for (const t of tables) {
      if (t[i] >= 0) {
        live[i] = 1;
        break;
      }
    }
  }
  if (startBoxes.some((b) => !live[b])) return null;

  // The matching estimate: cheapest assignment of boxes to goals over the push-distance tables.
  // With more boxes than a permutation table can cover, fall back to each box's nearest goal.
  const perms = n <= 5 ? permutations(n) : null;
  const hCache = new Map<string, number>();
  const estimate = (boxes: number[]): number => {
    const ck = boxes.join(',');
    const hit = hCache.get(ck);
    if (hit !== undefined) return hit;
    let best = Infinity;
    if (perms) {
      for (const p of perms) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const d = tables[p[i]][boxes[i]];
          if (d < 0) {
            sum = Infinity;
            break;
          }
          sum += d;
        }
        if (sum < best) best = sum;
      }
    } else {
      best = 0;
      for (const b of boxes) {
        let m = Infinity;
        for (const t of tables) if (t[b] >= 0 && t[b] < m) m = t[b];
        best += m;
      }
    }
    hCache.set(ck, best);
    return best;
  };

  // Scratch boards stamped per use, so nothing is reallocated inside the search: `occupied` marks
  // the box arrangement under one stamp; `seenHere` and `seenNext` hold the player's flooded
  // regions, two arrays because naming a push's new state floods again and must not erase the
  // region the expansion is still reading.
  const occupied = new Int32Array(size);
  const seenHere = new Int32Array(size);
  const seenNext = new Int32Array(size);
  let stamp = 0;
  const markBoxes = (boxes: number[]): number => {
    const s = ++stamp;
    for (const b of boxes) occupied[b] = s;
    return s;
  };
  const flood = (board: Int32Array, from: number, occS: number): { low: number; seenS: number } => {
    const s = ++stamp;
    let low = from;
    const stack = [from];
    board[from] = s;
    while (stack.length > 0) {
      const p = stack.pop() as number;
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k] * w + DX[k];
        if (q < 0 || q >= size) continue;
        if (wall[q] === 0 && occupied[q] !== occS && board[q] !== s) {
          board[q] = s;
          stack.push(q);
        }
      }
    }
    return { low, seenS: s };
  };
  const frozenBlock = (t: number, occS: number): boolean => {
    // A 2x2 of walls and boxes around the pushed box, with a box off its goal, can never move again.
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const cells = [a, a + 1, a + w, a + w + 1];
      let solid = true;
      let offGoal = false;
      for (const c of cells) {
        if (c < 0 || c >= size || !(wall[c] === 1 || occupied[c] === occS)) {
          solid = false;
          break;
        }
        if (occupied[c] === occS && !goalSet.has(c)) offGoal = true;
      }
      if (solid && offGoal) return true;
    }
    return false;
  };

  // States in creation order: their boxes, the player's square, parent, depth, and the push made.
  const stateBoxes: number[][] = [startBoxes.slice()];
  const statePlayer: number[] = [player];
  const stateParent: number[] = [-1];
  const stateDepth: number[] = [0];
  const stateFrom: number[] = [-1];
  const stateDir: number[] = [-1];
  const bestG = new Map<string, number>();
  const open = new OpenList();

  const h0 = estimate(startBoxes);
  if (h0 === Infinity) return null;
  const start = flood(seenHere, player, markBoxes(startBoxes));
  bestG.set(`${start.low}|${startBoxes.join(',')}`, 0);
  open.push(h0, 0, 0);

  let found = -1;
  let used = 0;
  while (open.size > 0) {
    const s = open.pop();
    const boxes = stateBoxes[s];
    if (boxes.every((b) => goalSet.has(b))) {
      found = s;
      break;
    }
    if (++used > cap || used > shared.left) {
      shared.left -= Math.min(used, shared.left);
      return null;
    }
    const occS = markBoxes(boxes);
    const region = flood(seenHere, statePlayer[s], occS);
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const stand = b - d;
        const dest = b + d;
        if (stand < 0 || stand >= size || dest < 0 || dest >= size) continue;
        if (seenHere[stand] !== region.seenS) continue; // the player cannot stand there
        if (wall[dest] === 1 || occupied[dest] === occS || !live[dest]) continue;
        occupied[b] = 0;
        occupied[dest] = occS;
        if (!frozenBlock(dest, occS)) {
          const next = boxes.slice();
          next[bi] = dest;
          next.sort((x, y) => x - y);
          const g2 = stateDepth[s] + 1;
          const key = `${flood(seenNext, b, occS).low}|${next.join(',')}`;
          const known = bestG.get(key);
          if (known === undefined || known > g2) {
            const h = estimate(next);
            if (h < Infinity) {
              bestG.set(key, g2);
              stateBoxes.push(next);
              statePlayer.push(b);
              stateParent.push(s);
              stateDepth.push(g2);
              stateFrom.push(b);
              stateDir.push(k);
              open.push(g2 + h, g2, stateBoxes.length - 1);
            }
          }
        }
        occupied[dest] = 0;
        occupied[b] = occS;
      }
    }
  }
  shared.left -= Math.min(used, shared.left);
  if (found < 0) return null;

  // The pushes, first to last; then the moves: walk to each push's stand square, then push.
  const chain: number[] = [];
  for (let s = found; s > 0; s = stateParent[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let at = player;
  const cur = startBoxes.slice();
  const occ = new Uint8Array(size);
  for (const s of chain) {
    const from = stateFrom[s];
    const k = stateDir[s];
    const d = DY[k] * w + DX[k];
    for (const b of cur) occ[b] = 1;
    moves += walkMoves(w, wall, occ, at, from - d);
    for (const b of cur) occ[b] = 0;
    moves += PUSH[k];
    cur[cur.indexOf(from)] = from + d;
    cur.sort((x, y) => x - y);
    at = from;
  }
  return moves;
}

/** The shortest walk from `from` to `to` around walls and boxes, lower-case LURD, ties by direction order. */
function walkMoves(w: number, wall: Uint8Array, occ: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Int32Array(size).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * w + DX[k];
      if (q < 0 || q >= size || wall[q] === 1 || occ[q] === 1 || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = STEP[how[p]] + s;
  return s;
}

/** Where the player ends after a move string, and the boxes it leaves, replayed on the plain board. */
function replay(level: Level, moves: string): { player: number; boxes: number[] } {
  const { w, wall } = level;
  let player = level.player;
  const boxes = new Set(level.boxes);
  for (const ch of moves.toLowerCase()) {
    const k = STEP.indexOf(ch);
    if (k < 0) continue;
    const d = DY[k] * w + DX[k];
    const next = player + d;
    if (next < 0 || next >= wall.length || wall[next] === 1) continue;
    if (boxes.has(next)) {
      const beyond = next + d;
      if (beyond < 0 || beyond >= wall.length || wall[beyond] === 1 || boxes.has(beyond)) continue;
      boxes.delete(next);
      boxes.add(beyond);
    }
    player = next;
  }
  return { player, boxes: [...boxes].sort((a, b) => a - b) };
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box there reaches no
 * goal) out of the floor; each part left holding boxes is a room, with its boxes and goals. Null
 * when the parts do not split the boxes and goals evenly, so the level is no set of rooms.
 */
function roomsOf(level: Level): { boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  const tables = pushDistances(w, wall, goals);
  const cut = new Uint8Array(size);
  const row = Math.floor(player / w);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (wall[i] === 1) continue;
    let liveCell = false;
    for (const t of tables) {
      if (t[i] >= 0) {
        liveCell = true;
        break;
      }
    }
    if (!liveCell) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b] === 1)) return null;
  const part = new Int32Array(size).fill(-1);
  const parts: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] === 1 || cut[s] === 1 || part[s] >= 0) continue;
    const id = parts.length;
    const squares = [s];
    part[s] = id;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i] + DY[k] * w + DX[k];
        if (q < 0 || q >= size) continue;
        if (wall[q] === 0 && cut[q] === 0 && part[q] < 0) {
          part[q] = id;
          squares.push(q);
        }
      }
    }
    parts.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) parts[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    parts[part[g]].goals.push(g);
  }
  if (parts.some((p) => p.boxes.length !== p.goals.length)) return null;
  return parts.filter((p) => p.boxes.length > 0);
}

/** One level's moves: room by room when it splits into rooms, the whole level otherwise. */
function solveLevel(text: string, shared: { left: number }): string {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (!rooms) {
    return solvePushes(level.w, level.wall, level.goals, level.boxes, level.player, LEVEL_BUDGET, shared) ?? '';
  }
  let moves = '';
  const done = new Array<boolean>(rooms.length).fill(false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const wall = level.wall.slice();
    rooms.forEach((other, oi) => {
      if (oi === ri) return;
      for (const c of done[oi] ? other.goals : other.boxes) wall[c] = 1;
    });
    const playerNow = replay(level, moves).player;
    const solved = solvePushes(level.w, wall, rooms[ri].goals, rooms[ri].boxes, playerNow, ROOM_BUDGET, shared);
    if (solved === null) continue; // this room resists: its boxes stay put and the rest still solve
    moves += solved;
    done[ri] = true;
  }
  return moves;
}

const shared = { left: 2_500_000 };
for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'), shared);
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
