from setuptools import setup
from glob import glob

setup(name='lightfield_litenet', version='0.1.0', packages=['lfnav'],
      data_files=[
          ('share/ament_index/resource_index/packages', ['resource/lightfield_litenet']),
          ('share/lightfield_litenet', ['package.xml', 'README.md', 'LICENSE']),
          ('share/lightfield_litenet/launch', glob('launch/*.launch.py')),
          ('share/lightfield_litenet/configs', glob('configs/ros_*.json')),
          ('share/lightfield_litenet/weights', ['weights/best.onnx'])],
      install_requires=['setuptools', 'numpy'], zip_safe=False,
      maintainer='Lightfield LiteNet contributors', maintainer_email='maintainer@example.com',
      description='3x3 light field ONNX depth inference', license='MIT',
      entry_points={'console_scripts': ['depth_node = lfnav.ros_node:main']})
