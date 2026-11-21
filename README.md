# Native Push — a room-by-room push-optimal Sokoban solver

`main.ts` (Node 22, no dependencies) answers a set of Sokoban levels: it reads
`/input/<id>/level.txt` (XSB) and writes the player's moves to
`/output/<id>/solution.txt`, in as few pushes as possible.

## How it works

The levels are built from independent rooms off a one-wide hallway, and a box
can never leave its room — pushed through the door it would sit in the hallway,
from where no goal is reachable. So a level's push optimum is the sum of its
rooms' optima, and the solver works room by room:

1. **Split.** Cut the dead squares of the player's row (the hallway) out of the
   floor; each part left holding boxes is a room.
2. **Solve each room push-optimally** with an A* search over pushes. A state is
   the sorted box squares plus the player's reachable region, so walks between
   pushes are never part of the state. The estimate is a minimum-cost matching
   of boxes to goals over true push distances (a lone box's cheapest pushes to
   each goal, other boxes ignored), which never overstates the pushes still
   needed — so the first solution found is push-optimal. Pushes landing a box
   on a dead square, or freezing a 2x2 block of walls and boxes around an
   off-goal box, are pruned.
3. **Carry the player on** from where each room's solution leaves them, the
   other rooms' boxes standing as walls, and concatenate the moves.

All budgets are counts of expanded states, never the clock, and every tie is
broken by square order, so two runs of the program write byte-identical
answers. On the kit's practice set it scores the maximum: every instance
solved in exactly the reference push count.
