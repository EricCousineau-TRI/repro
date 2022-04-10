#!/usr/bin/env python3

def generate_sdf(model, poses_file, random, file_name):
  sdf_text = '''<sdf version="1.6">
  <world name="default">
      <plugin
      filename="ignition-gazebo-physics-system"
      name="ignition::gazebo::systems::Physics">
      </plugin>
      <plugin
      filename="ignition-gazebo-sensors-system"
      name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
      <background_color>0, 1, 0</background_color>
      </plugin>
      <plugin
          filename="ignition-gazebo-user-commands-system"
          name="ignition::gazebo::systems::UserCommands">
      </plugin>
      <plugin
      filename="ignition-gazebo-scene-broadcaster-system"
      name="ignition::gazebo::systems::SceneBroadcaster">
      </plugin>
      <include>
      <uri>''' + model + '''</uri>
      <plugin
          filename="ignition-gazebo-model-photo-shoot-system"
          name="ignition::gazebo::systems::ModelPhotoShoot">
          <translation_data_file>''' + poses_file + '''</translation_data_file>
          <random_joints_pose>''' + random + '''</random_joints_pose>
      </plugin>
      </include>
      <model name="photo_shoot">
      <pose>2.2 0 0 0 0 -3.14</pose>
      <link name="link">
          <pose>0 0 0 0 0 0</pose>
          <sensor name="camera" type="camera">
          <camera>
              <horizontal_fov>1.047</horizontal_fov>
              <image>
              <width>960</width>
              <height>540</height>
              </image>
              <clip>
              <near>0.1</near>
              <far>100</far>
              </clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>30</update_rate>
          <visualize>true</visualize>
          <topic>camera</topic>
          </sensor>
      </link>
      <static>true</static>
      </model>
  </world>
</sdf>'''
  with open(file_name, 'w') as f:
    f.write(sdf_text)

