from security_drone.config import load_config


def test_default_config_has_trigger_seq():
    config = load_config()
    assert isinstance(config["mission"]["trigger_seq"], int)
    assert config["mavlink"]["connection"].startswith("tcp:")
