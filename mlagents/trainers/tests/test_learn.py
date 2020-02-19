import pytest
from unittest.mock import MagicMock, patch
from mlagents.trainers import learn
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.learn import parse_command_line


@pytest.fixture
def basic_options(extra_args=None):
    extra_args = extra_args or {}
    args = ["basic_path"]
    if extra_args:
        args += [f"{k}={v}" for k, v in extra_args.items()]
    return parse_command_line(args)


@patch("mlagents.trainers.learn.SamplerManager")
@patch("mlagents.trainers.learn.SubprocessEnvManager")
@patch("mlagents.trainers.learn.create_environment_factory")
@patch("mlagents.trainers.learn.load_config")
def test_run_training(
    load_config, create_environment_factory, subproc_env_mock, sampler_manager_mock
):
    mock_env = MagicMock()
    mock_env.external_brain_names = []
    mock_env.academy_name = "TestAcademyName"
    create_environment_factory.return_value = mock_env
    trainer_config_mock = MagicMock()
    load_config.return_value = trainer_config_mock

    mock_init = MagicMock(return_value=None)
    with patch.object(TrainerController, "__init__", mock_init):
        with patch.object(TrainerController, "start_learning", MagicMock()):
            learn.run_training(0, 0, basic_options(), MagicMock())
            mock_init.assert_called_once_with(
                {},
                "./models/ppo-0",
                "./summaries",
                "ppo-0",
                50000,
                None,
                False,
                0,
                True,
                sampler_manager_mock.return_value,
                None,
            )


@patch("mlagents.trainers.learn.SamplerManager")
@patch("mlagents.trainers.learn.SubprocessEnvManager")
@patch("mlagents.trainers.learn.create_environment_factory")
@patch("mlagents.trainers.learn.load_config")
def test_docker_target_path(
    load_config, create_environment_factory, subproc_env_mock, sampler_manager_mock
):
    mock_env = MagicMock()
    mock_env.external_brain_names = []
    mock_env.academy_name = "TestAcademyName"
    create_environment_factory.return_value = mock_env
    trainer_config_mock = MagicMock()
    load_config.return_value = trainer_config_mock

    options_with_docker_target = basic_options({"--docker-target-name": "dockertarget"})

    mock_init = MagicMock(return_value=None)
    with patch.object(TrainerController, "__init__", mock_init):
        with patch.object(TrainerController, "start_learning", MagicMock()):
            learn.run_training(0, 0, options_with_docker_target, MagicMock())
            mock_init.assert_called_once()
            assert mock_init.call_args[0][1] == "/dockertarget/models/ppo-0"
            assert mock_init.call_args[0][2] == "/dockertarget/summaries"


def test_commandline_args():

    # No args raises
    with pytest.raises(SystemExit):
        parse_command_line([])

    # Test with defaults
    opt = parse_command_line(["mytrainerpath"])
    assert opt.trainer_config_path == "mytrainerpath"
    assert opt.env_path is None
    assert opt.curriculum_folder is None
    assert opt.sampler_file_path is None
    assert opt.keep_checkpoints == 5
    assert opt.lesson == 0
    assert opt.load_model is False
    assert opt.run_id == "ppo"
    assert opt.save_freq == 50000
    assert opt.seed == -1
    assert opt.fast_simulation is True
    assert opt.train_model is False
    assert opt.base_port == 5005
    assert opt.num_envs == 1
    assert opt.docker_target_name is None
    assert opt.no_graphics is False
    assert opt.debug is False
    assert opt.multi_gpu is False
    assert opt.env_args is None

    full_args = [
        "mytrainerpath",
        "--env=./myenvfile",
        "--curriculum=./mycurriculum",
        "--sampler=./mysample",
        "--keep-checkpoints=42",
        "--lesson=3",
        "--load",
        "--run-id=myawesomerun",
        "--num-runs=3",
        "--save-freq=123456",
        "--seed=7890",
        "--slow",
        "--train",
        "--base-port=4004",
        "--num-envs=2",
        "--docker-target-name=mydockertarget",
        "--no-graphics",
        "--debug",
        "--multi-gpu",
    ]

    opt = parse_command_line(full_args)
    assert opt.trainer_config_path == "mytrainerpath"
    assert opt.env_path == "./myenvfile"
    assert opt.curriculum_folder == "./mycurriculum"
    assert opt.sampler_file_path == "./mysample"
    assert opt.keep_checkpoints == 42
    assert opt.lesson == 3
    assert opt.load_model is True
    assert opt.run_id == "myawesomerun"
    assert opt.save_freq == 123456
    assert opt.seed == 7890
    assert opt.fast_simulation is False
    assert opt.train_model is True
    assert opt.base_port == 4004
    assert opt.num_envs == 2
    assert opt.docker_target_name == "mydockertarget"
    assert opt.no_graphics is True
    assert opt.debug is True
    assert opt.multi_gpu is True


def test_env_args():
    full_args = [
        "mytrainerpath",
        "--env=./myenvfile",
        "--env-args",  # Everything after here will be grouped in a list
        "--foo=bar",
        "--blah",
        "baz",
        "100",
    ]

    opt = parse_command_line(full_args)
    assert opt.env_args == ["--foo=bar", "--blah", "baz", "100"]
