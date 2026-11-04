"""Budget the long phase, including its hour evaluation and a 35-minute final reserve."""
def make_plan(gpu_used,training_seconds,anchor,evaluation_pair_minutes,model):
    available=300-gpu_used-35-8-2
    if available<=0:raise ValueError('No training budget after final-evaluation reserve')
    hour_cost=2.2*evaluation_pair_minutes
    shared=False
    if available>=60+hour_cost:
        duration=available-hour_cost
    elif available>=60:
        # One checkpoint at 60 minutes serves both requested evaluation labels.
        # Running identical seeds twice would not add independent evidence.
        duration=60.;shared=True
    else:
        duration=available;hour_cost=0.
    start=training_seconds/60
    return dict(candidate_model=model,start_training_minutes=start,end_training_minutes=start+duration,additional_training_minutes=duration,gpu_used_at_plan=gpu_used,final_evaluation_reserve=35,desynchronization_reserve=8,overhead_reserve=2,hour_evaluation_estimate=hour_cost,anchor_start=anchor,shared_hour_final=shared,evaluation_pair_minutes=evaluation_pair_minutes,planning_rule='Reserve 35 final + 8 desynchronization + 2 overhead; estimate hour evaluation as 2.2x selected pilot v10 pair. If only about one hour fits, one evaluation at the 60-minute final checkpoint serves both labels; do not count it as two independent series.')
