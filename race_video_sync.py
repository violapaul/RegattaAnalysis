#!/usr/bin/env python
# Process the virb video and fit files to sync and compute poses
import os
import utils

from global_variables import G
import virb_video_pose
G.init_seattle()

description = [
    "Process video and log data to sync and extract."
]

description = "\n".join(description)

if __name__ == "__main__":
    # G.set_logging_level(args.log)
    # G.logger.info(f"Set log level to {args.log}")

    virb_video_pose.find_video_poses_logs()
    virb_video_pose.convert_logs()
