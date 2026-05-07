---
title: "How would you debug a model that's not learning?"
description: "The 'tell me how you'd debug' question is a behavioral round in disguise. The interviewer is probing your debugging instinct, not testing facts."
date: "2026-05-07"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: ML breadth or behavioral rounds at every level.*

The interviewer is checking debugging instinct, not facts. The L4 candidate gives a list of things to check. The L6 candidate describes a procedure ordered by cost and information value.

## What an L4 answer sounds like

> "I'd check the learning rate. Maybe try a smaller one. I'd also check if there's a bug in the data loader or the loss function. Make sure the model isn't stuck at a local minimum."

This is a list of things, in no particular order, each generic. You've heard the canonical advice but haven't internalized a method.

## What an L5 answer sounds like

> "The way I'd actually debug this is: top-down, fastest-checks-first.
>
> **First, can the model overfit a single batch?** Take 8 examples, train on them for a few hundred steps. If loss doesn't go to ~zero, the problem is fundamental, broken loss, broken gradient flow, broken data pipeline, model architecture can't represent the function. Fix that before anything else.
>
> **Second, are the gradients reasonable?** Check gradient norms per layer. If they're zero or NaN, look for non-differentiable ops, broken initialization, or numerical issues. If they're huge, learning rate is too high or there's an exploding gradient.
>
> **Third, is the loss going down at all?** If yes but slowly, it's an LR or optimizer issue. Try an LR sweep across 4 orders of magnitude. If no, go back to step 1, the model genuinely isn't training.
>
> **Fourth, is the validation loss diverging from training?** If yes, you've found the right problem and the issue is overfitting. Add regularization, more data, or augmentation.
>
> **Fifth, is the validation loss tracking training but neither is good enough?** Then you're underfitting, you need more capacity, more data, or a better representation."

This is L5. You've described a *procedure* (top-down, fast checks first), with specific tests at each stage and a clear branch on what each result implies.

## What an L6 answer sounds like

The L6 answer adds the things that come from having debugged a hundred models:

> "...and a few things I've learned to check that aren't obvious:
>
> **Are the labels correct?** I once spent two weeks debugging a model that wasn't learning, only to discover the data team had silently flipped a label convention. Now I always print 5-10 random (input, label) pairs and *verify visually that the labels make sense*. This catches data pipeline bugs that no metric will.
>
> **Is the loss function actually what I think it is?** Compute the loss on a single example by hand and compare to what the code outputs. Common bugs: reduction='sum' vs 'mean' silently changing the effective learning rate, ignored class weights, broken masking on padding tokens.
>
> **Is the model in train mode?** It sounds dumb. I've seen models 'not learn' because someone left them in eval mode and dropout / BN were behaving wrong. Check `model.train()` is called before the train loop.
>
> **Is the data shuffled?** If you forgot to shuffle, the model sees N batches of all-class-0 followed by N batches of all-class-1 and the optimizer will do something weird.
>
> **Are you running on the GPU?** Specifically, is the data on the GPU? If you're moving tensors back and forth between CPU and GPU on every step, training will be 100&times; slower than it should be and look like it's not learning when it actually just hasn't done enough steps yet.
>
> **What's your effective batch size?** If you have gradient accumulation set up wrong, your effective batch can be 8 instead of 2048 and the optimizer dynamics look completely different.
>
> **Is the training distribution shifted?** If your eval set has a class your training set doesn't, no amount of training will help.
>
> A useful diagnostic that catches most issues: print the first prediction the model makes on a fixed batch every 100 steps. Watch how it evolves. If it's stuck at the same value, you have a gradient problem. If it's chaotic, you have an LR problem. If it's slowly improving, you have a patience problem. If it's improving on training but not validation, you have an overfitting problem."

This is L6. You've gone past the standard checklist into the things that come from real failure stories. Notice the *specific bug stories*: this is what makes the answer feel earned rather than rehearsed.

## The tells that get you a strong-hire vote

- You describe a **procedure** (overfit-a-batch first, then gradients, then LR sweep), not a list.
- You give **specific bug stories**: "I once spent two weeks because..."
- You mention **looking at predictions, not just losses**: the most underrated debugging move.
- You think about **data quality** as a debugging axis, not just model code.

## The tells that get you down-leveled

- You give an unordered list of things to check.
- You don't mention overfit-a-single-batch, the canonical first check.
- You say "try Adam instead of SGD" or other surface-level model swaps as if those are debugging.
- You can't describe a real model you debugged and what was wrong.

## A follow-up the interviewer often asks

"What if the loss is going down but slowly?" or "what if the loss looks fine but the model is bad at the task?"

The first is an LR / capacity question; the second is a question about the *eval-vs-task gap* and is often more interesting. The L6 answer to the second:

> "If the loss looks fine but the model is bad, the loss isn't measuring what I care about. Either the loss function is wrong (e.g., MSE on classification), or the data distribution doesn't reflect the use case, or there's a labeling issue, or the eval metric is wrong. I'd construct a small set of failure cases by hand, compute loss on them, and see if low-loss-but-bad cases exist, that's the disconnect."

If you have this conversation fluently, you're at the senior bar.

---

*Related: [How would you evaluate an LLM application?](/interviews/how-would-you-evaluate-an-llm-application/), [Walk me through bias-variance tradeoff](/interviews/bias-variance-tradeoff/).*
