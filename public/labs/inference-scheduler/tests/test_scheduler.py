import unittest

from scheduler import Request, Scheduler


class SchedulerTests(unittest.TestCase):
    def test_rejects_request_that_exceeds_kv_capacity(self) -> None:
        scheduler = Scheduler(kv_blocks=4, block_size=16)
        accepted = scheduler.submit(Request("large", "a", prompt_tokens=48, max_new_tokens=32))
        self.assertFalse(accepted)
        self.assertEqual(scheduler.snapshot()["rejected"], 1)

    def test_reserves_and_releases_kv_blocks(self) -> None:
        scheduler = Scheduler(kv_blocks=16, block_size=16, max_batch_tokens=16, prefill_chunk=16)
        request = Request("r1", "a", prompt_tokens=16, max_new_tokens=16)
        self.assertTrue(scheduler.submit(request))
        self.assertEqual(scheduler.used_blocks, 2)
        work = scheduler.next_batch()
        self.assertEqual(work, [("r1", "prefill", 16)])
        work = scheduler.next_batch()
        self.assertEqual(work, [("r1", "decode", 1)])
        scheduler.complete_step("r1", emitted_eos=True)
        self.assertEqual(scheduler.used_blocks, 0)

    def test_chunks_long_prefill_and_preserves_decode(self) -> None:
        scheduler = Scheduler(kv_blocks=64, block_size=16, max_batch_tokens=9, prefill_chunk=8)
        scheduler.submit(Request("long", "a", prompt_tokens=40, max_new_tokens=4))
        scheduler.submit(Request("short", "b", prompt_tokens=1, max_new_tokens=4))
        first = scheduler.next_batch()
        self.assertLessEqual(sum(item[2] for item in first), 9)
        second = scheduler.next_batch()
        phases = {request_id: phase for request_id, phase, _ in second}
        self.assertEqual(phases.get("short"), "decode")
        self.assertLessEqual(sum(item[2] for item in second), 9)

    def test_rotates_across_tenants(self) -> None:
        scheduler = Scheduler(kv_blocks=64, block_size=16, max_batch_tokens=2, prefill_chunk=1)
        for index in range(3):
            scheduler.submit(Request(f"a{index}", "a", prompt_tokens=1, max_new_tokens=1))
        scheduler.submit(Request("b0", "b", prompt_tokens=1, max_new_tokens=1))
        work = scheduler.next_batch()
        tenants_by_id = {request.request_id: request.tenant for queue in scheduler.queues.values() for request in queue}
        tenants_by_id.update({request.request_id: request.tenant for request in scheduler.active.values()})
        self.assertEqual({tenants_by_id[request_id] for request_id, _, _ in work}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
