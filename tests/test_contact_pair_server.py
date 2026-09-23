import pytest
import torch
from test_dense_kv_index_server import dense_server
from n0_twam.preprocessing.kv_index import observed_index


def inputs():
    server = dense_server()
    server.job_config.obs_cam_keys = ['observation.images.top', 'observation.images.wrist_l']
    server.job_config.tactile_keys = ['left', 'right']
    server.job_config.patch_size = (1, 2, 2)
    visual = torch.zeros(1, 1, 2, 2, 4)  # F2, H1, two camera columns
    tactile = {'tactile_global_latent': torch.zeros(1, 2, 1, 2, 2, 2)}
    index = observed_index({'dino': torch.ones(4, 3)}, 4, 'cpu')
    forward = {'latent_res_lst': {}, 'action_res_lst': {}}
    pairs = {'neoforce': torch.arange(8.).reshape(4, 2) + 1,
             'response': torch.ones(4), 'visual_rows': torch.tensor([0, 2, 1, -1])}
    return server, forward, index, visual, tactile, pairs


def test_grounding_routes_into_video_and_both_existing_tactile_tails():
    server, forward, index, visual, tactile, pairs = inputs()
    server._attach_observed_contact_pairs({'contact_pairs': pairs}, forward, index, visual, tactile)
    assert forward['latent_res_lst']['kv_index']['neoforce'].tolist() == [[1, 2], [0, 0], [3, 4], [0, 0]]
    assert torch.equal(forward['latent_res_lst']['kv_index']['dino'], index['dino'])
    tail = forward['latent_res_lst']['tactile_kv_index']
    assert tail['neoforce'].tolist() == [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert tail['dino'].tolist() == [[1, 1, 1], [1, 1, 1],
                                     [0, 0, 0], [0, 0, 0]]
    assert forward['action_res_lst']['tactile_kv_index'] is tail
    assert index['neoforce'].shape == (4, 0)  # input metadata never modified


@pytest.mark.parametrize('problem', ['time', 'count', 'conflict'])
def test_invalid_contact_pairs_fail_before_forward_input_mutation(problem):
    server, forward, index, visual, tactile, pairs = inputs()
    obs = {'contact_pairs': pairs}
    if problem == 'time':
        pairs['visual_rows'][0] = 2
    elif problem == 'count':
        pairs['neoforce'] = pairs['neoforce'][:1]
    else:
        obs['tactile_kv_index'] = {}
    with pytest.raises(ValueError):
        server._attach_observed_contact_pairs(obs, forward, index, visual, tactile)
    assert forward == {'latent_res_lst': {}, 'action_res_lst': {}}
