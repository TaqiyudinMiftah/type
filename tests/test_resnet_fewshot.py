"""Check transfer state, paired randomness and exact optimizer/RNG recovery."""

import torch

from ketik_audio.resnet_fewshot import (
    cpu_state,
    initialize_model,
    load_recovery,
    save_recovery,
    seed_training,
    state_digest,
    training_batch,
)


def test_source_initialization_copies_head_and_bn_without_aliasing_source():
    torch.set_num_threads(2)
    source = initialize_model(43)
    with torch.no_grad():
        source[3].bias.add_(1)
        source[0].s0[1].running_mean.fill_(0.5)
        source[0].s0[1].num_batches_tracked.fill_(12)
    saved = cpu_state(source)
    expected = state_digest(saved)
    adapted = initialize_model(42, saved)
    assert state_digest(adapted.state_dict()) == expected
    assert state_digest(initialize_model(42).state_dict()) != expected
    with torch.no_grad():
        adapted[3].bias.add_(2)
        adapted[0].s0[1].running_mean.add_(1)
    assert state_digest(saved) == expected
    assert state_digest(source.state_dict()) == expected


def test_training_batches_and_augmentations_match_despite_different_initial_weights():
    torch.set_num_threads(2)
    torch.manual_seed(999)
    wave, labels = torch.randn(72, 6400), torch.arange(36).repeat(2)
    mean, std = torch.zeros(1, 64, 1), torch.ones(1, 64, 1)
    source = cpu_state(initialize_model(43))
    traces = []
    for source_state in (None, source):
        model = initialize_model(42, source_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        assert len(optimizer.state) == 0
        seed_training(42)
        batches = []
        for _ in range(2):
            ix, images, y = training_batch(wave, labels, mean, std)
            batches.append((ix.clone(), images.clone(), y.clone()))
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(images), y).backward()
            optimizer.step()
        traces.append(batches)
    for scratch, adapted in zip(*traces, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(scratch, adapted, strict=True))


def test_recovery_restores_optimizer_scheduler_and_random_stream(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(123)

    def setup():
        model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0, total_iters=10_000
        )
        return model, optimizer, scheduler

    def update(model, optimizer, scheduler):
        x, y = torch.randn(4, 6), torch.randint(3, (4,))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return x, float(loss.item())

    model, optimizer, scheduler = setup()
    update(model, optimizer, scheduler)
    save_recovery(tmp_path / "recovery.pt", model, optimizer, scheduler, 1, {"marker": 5})
    expected_x, expected_loss = update(model, optimizer, scheduler)
    expected_state = state_digest(model.state_dict())
    restored, opt2, sched2 = setup()
    step, metadata = load_recovery(tmp_path / "recovery.pt", restored, opt2, sched2)
    assert step == 1 and metadata == {"marker": 5}
    actual_x, actual_loss = update(restored, opt2, sched2)
    assert torch.equal(actual_x, expected_x)
    assert actual_loss == expected_loss
    assert state_digest(restored.state_dict()) == expected_state
    assert opt2.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
