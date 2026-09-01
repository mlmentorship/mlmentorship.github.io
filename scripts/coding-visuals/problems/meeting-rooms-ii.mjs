import { defineVisual, frame, heap, visual } from '../primitives.mjs';

const example = 'intervals = [[0,30], [5,10], [5,15], [20,25]]';
const state = (values, extra = {}) => heap(values, { example, heapMeaning: 'active meeting end times (minimum at root)', ...extra });

const draft = visual('Before each start, pop every ended meeting; then push the new end and record the largest heap size.', [
  frame(
    'Initialize the active heap',
    'Meetings are sorted by start as [0,30], [5,10], [5,15], [20,25]. end_times is empty and most_rooms=0.',
    state(['empty'], { nextMeeting: '[0,30]', mostRooms: '0' }),
    'initialize-active-heap',
  ),
  frame(
    'Push end 30',
    'At start 0 the heap is empty, so no meeting can be popped. Push 30; heap size 1 raises most_rooms to 1.',
    state(['30'], { meeting: '[0,30]', operation: 'push 30', heapSize: '1', mostRooms: '1' }),
    'push-thirty',
  ),
  frame(
    'Push end 10',
    'At start 5, minimum end 30 <= 5 is false. Push 10, which becomes the min-heap root; most_rooms becomes 2.',
    state(['10', '30'], { meeting: '[5,10]', comparison: '30 <= 5: false', operation: 'push 10', heapSize: '2', mostRooms: '2' }),
    'push-ten',
  ),
  frame(
    'Push end 15',
    'The next meeting also starts at 5. Minimum end 10 <= 5 is false, so push 15; three meetings are active.',
    state(['10', '30', '15'], { meeting: '[5,15]', comparison: '10 <= 5: false', operation: 'push 15', heapSize: '3', mostRooms: '3' }),
    'push-fifteen',
  ),
  frame(
    'Pop end 10',
    'Before [20,25], minimum end 10 <= start 20 is true. Pop 10; the heap becomes [15,30].',
    state(['15', '30'], { meeting: '[20,25]', comparison: '10 <= 20: true', operation: 'pop 10', heapSize: '2', mostRooms: '3' }),
    'pop-ten',
  ),
  frame(
    'Pop end 15',
    'The while loop checks again: minimum end 15 <= 20 is true. Pop 15; only end 30 remains active.',
    state(['30'], { meeting: '[20,25]', comparison: '15 <= 20: true', operation: 'pop 15', heapSize: '1', mostRooms: '3' }),
    'pop-fifteen',
  ),
  frame(
    'Stop popping at end 30',
    'The next root 30 <= 20 is false, so [0,30] still occupies a room and the cleanup loop stops.',
    state(['30'], { meeting: '[20,25]', comparison: '30 <= 20: false', operation: 'stop cleanup', heapSize: '1', mostRooms: '3' }),
    'stop-cleanup',
  ),
  frame(
    'Push end 25',
    'Push the new end 25, which becomes the root above 30. Heap size is 2, so most_rooms stays 3.',
    state(['25', '30'], { meeting: '[20,25]', operation: 'push 25', heapSize: '2', update: 'max(3, 2) = 3', mostRooms: '3' }),
    'push-twenty-five',
  ),
  frame(
    'Return the peak heap size',
    'The active heap reached size 3 when [0,30], [5,10], and [5,15] overlapped, so three rooms are necessary and sufficient.',
    state(['25', '30'], { peakOverlap: '[0,30], [5,10], [5,15]', result: '3' }),
    'return-three-rooms',
  ),
]);

const review = {
  pattern: 'Sweep meetings by start time while a min-heap stores end times of currently active meetings.',
  recognitionCue: 'The question asks for the maximum number of simultaneous intervals, and meetings that end before the next start release reusable capacity.',
  invariant: 'Immediately before pushing a meeting, the heap contains exactly the end times greater than its start; after pushing, heap size equals active rooms and most_rooms is the largest size seen.',
  stateModel: 'Retain the start-sorted meetings, min-heap of active end times, current meeting, and scalar most_rooms. Room identities and a full timeline are unnecessary.',
  visualRationale: 'A real binary min-heap makes the earliest releasable room the root and shows every repeated pop before the next end is pushed.',
  rejectedAlternatives: [
    'A timeline alone shows overlap but hides why a min-heap can release every ended room efficiently.',
    'Separate sorted start/end pointers solve the problem differently and would not match the supplied heap implementation.',
    'A room-assignment table invents room identities that the algorithm never stores.',
  ],
  transferLesson: 'For capacity over time, sweep arrivals and keep active expirations ordered by the next one to finish; this transfers to server concurrency, platform usage, and resource reservation.',
  reviewStatus: 'reviewed',
};

export default defineVisual('meeting-rooms-ii', draft, review);
