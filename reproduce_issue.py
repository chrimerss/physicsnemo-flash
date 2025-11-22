from omegaconf import OmegaConf
import hydra
import json

def test_omegaconf_conversion():
    # Simulate the config
    yaml_conf = """
    net:
      _target_: test_class
      img_resolution: [1024, 1024]
      channels: 20
    """
    cfg = OmegaConf.create(yaml_conf)
    
    print(f"Original config type: {type(cfg.net)}")
    print(f"Original img_resolution type: {type(cfg.net.img_resolution)}")
    
    # Convert to container
    net_cfg = OmegaConf.to_container(cfg.net, resolve=True)
    
    print(f"Converted config type: {type(net_cfg)}")
    print(f"Converted img_resolution type: {type(net_cfg['img_resolution'])}")
    
    # Verify JSON serialization
    try:
        json.dumps(net_cfg)
        print("JSON serialization successful!")
    except TypeError as e:
        print(f"JSON serialization failed: {e}")

if __name__ == "__main__":
    test_omegaconf_conversion()
