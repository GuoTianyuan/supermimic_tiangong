"""Reward units, episode resets/weighting, resume CSV and real event files."""
import csv
import math
from types import SimpleNamespace
import torch
import pytest
from supermimic.logging.episode import EpisodeMetrics
from supermimic.logging.schema import REWARD_NAMES, dashboard_scalars
from supermimic.logging.logger import TrainingLogger, open_csv, LOG_FIELDS
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def transition(done, steps, values):
    values=torch.tensor(values,dtype=torch.float64)
    return dict(terms={k:values for k in REWARD_NAMES},r=values*len(REWARD_NAMES),
                done=torch.tensor(done),ep_step=torch.tensor(steps))


def test_episode_reset_flush_and_no_rng_or_input_mutation():
    m=EpisodeMetrics(2,5.,'cpu');rng=torch.get_rng_state().clone()
    out=transition([True,False],[0,0],[1.,2.]);before=out['r'].clone()
    m.update(out); a=m.flush(.01)
    assert a['episode_reward_body_pos']==.2
    assert a['episode_return']==9
    assert a['episode_length_seconds']==.01
    assert m.flush()=={}
    # env 1 continues across flushes, env 0 starts a fresh episode.
    m.update(transition([True,True],[0,1],[3.,4.]))
    b=m.flush(.01)
    assert b['episode_reward_body_pos']==pytest.approx(.9) # (3 + (2+4))/2/5
    assert b['episode_length_seconds']==.015
    assert b['episode_count']==2
    assert torch.equal(out['r'],before) and torch.equal(torch.get_rng_state(),rng)
    assert not m.running.any()


def test_rank_counts_weight_episodes_not_rank_means():
    a,b=EpisodeMetrics(1,5.,'cpu'),EpisodeMetrics(2,5.,'cpu')
    a.update(transition([True],[0],[1.]))
    b.update(transition([True,True],[0,0],[2.,9.]))
    a.completed.add_(b.completed)
    assert a.flush()['episode_reward_body_pos']==pytest.approx(12/3/5)


def test_reward_units_and_historical_missing_episodes():
    r=dict(reward_body_pos=.008,pen_joint_acc=-.003,sim_reward_normalized=.7,wm_err8=.12)
    t=dashboard_scalars(r,.01,12)
    assert t['Reward/body_position']==.8
    assert t['Reward/joint_acceleration']==-.3
    assert t['Reward/total']==3.5
    assert t['World_Model/replay_error_12_steps']==.12
    assert not any(k.startswith('Episode_Reward/') for k in t)
    with pytest.raises(ValueError):dashboard_scalars(r,0,12)


def test_csv_resume_keeps_old_units_columns_and_prunes_newer_steps(tmp_path):
    path=tmp_path/'train_log.csv'
    path.write_text('iteration,reward_body_pos,wm_loss\n50,0.008,0.1\n100,0.009,0.2\n150,0.01,0.3\n')
    stream,writer=open_csv(path,100);stream.close()
    rows=list(csv.DictReader(path.open()))
    assert [r['iteration'] for r in rows]==['50','100']
    assert rows[0]['reward_body_pos']=='0.008'
    assert math.isnan(float(rows[0]['episode_reward_body_pos']))
    assert list(rows[0])==LOG_FIELDS
    with pytest.raises(FileExistsError):open_csv(path,0)


def test_event_files_separate_diagnostics_and_can_be_read(tmp_path):
    args=SimpleNamespace(log_dir=str(tmp_path),wm_window=12,iterations=100)
    logger=TrainingLogger(args)
    row=dict(iteration=50,wm_loss=.1,wm_err8=.2,reward_body_pos=.009,sim_reward_normalized=.8)
    logger.log(row,{'2_world_model/channels/test/raw_rmse':.123},{'episode_reward_body_pos':.4},
               steps_per_s=10,elapsed_s=1,eta_s=1,last_ckpt='none')
    logger.close();logger.close()
    main=EventAccumulator(str(tmp_path/'tensorboard')).Reload()
    detail=EventAccumulator(str(tmp_path/'tensorboard_diagnostics')).Reload()
    assert main.Scalars('Reward/body_position')[0].value==pytest.approx(.9)
    assert main.Scalars('Episode_Reward/body_position')[0].value==pytest.approx(.4)
    assert '2_world_model/channels/test/raw_rmse' not in main.Tags()['scalars']
    assert detail.Scalars('2_world_model/channels/test/raw_rmse')[0].value==pytest.approx(.123)


def test_resume_old_dashboard_archives_channels_and_rebuilds_readable_history(tmp_path):
    from torch.utils.tensorboard import SummaryWriter
    (tmp_path/'train_log.csv').write_text('iteration,reward_body_pos,wm_loss\n50,0.008,0.1\n100,0.009,0.2\n')
    with SummaryWriter(str(tmp_path/'tensorboard')) as old:
        old.add_scalar('2_world_model/channels/old',42,50)
    args=SimpleNamespace(log_dir=str(tmp_path),wm_window=12,iterations=100)
    logger=TrainingLogger(args,start_it=100);logger.close()
    main=EventAccumulator(str(tmp_path/'tensorboard')).Reload()
    old=EventAccumulator(str(tmp_path/'tensorboard_legacy')).Reload()
    assert [x.step for x in main.Scalars('Reward/body_position')]==[50,100]
    assert '2_world_model/channels/old' not in main.Tags()['scalars']
    assert old.Scalars('2_world_model/channels/old')[0].value==42
    assert not any(k.startswith('Episode_Reward/') for k in main.Tags()['scalars'])
