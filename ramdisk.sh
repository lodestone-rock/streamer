#!/bin/bash

# Function to create and mount the RAM disk
create_ramdisk() {
  # Check if the mount point exists
  if [ ! -d "$mount_point" ]; then
    echo "Creating the mount point directory..."
    mkdir -p "$mount_point"
  fi

  # Calculate the size in megabytes (1 GB = 1024 MB)
  ramdisk_size_mb=$((ramdisk_size_gb * 1024))

  # Create and mount the RAM disk
  echo "Creating a ${ramdisk_size_gb}GB RAM disk and mounting it to $mount_point..."
  sudo mount -t tmpfs -o size=${ramdisk_size_mb}M tmpfs "$mount_point"

  if [ $? -eq 0 ]; then
    echo "RAM disk created and mounted successfully."
  else
    echo "Failed to create and mount the RAM disk."
    exit 1
  fi
}

# Function to unmount the RAM disk
unmount_ramdisk() {
  echo "Unmounting the RAM disk at $mount_point..."
  sudo umount "$mount_point"

  if [ $? -eq 0 ]; then
    echo "RAM disk unmounted successfully."
  else
    echo "Failed to unmount the RAM disk."
    exit 1
  fi
}

# Check if the script is run with the correct number of arguments
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <ramdisk_size_in_GB> <mount_point> [unmount]"
  exit 1
fi

ramdisk_size_gb="$1"
mount_point="$2"

if [ "$#" -eq 3 ] && [ "$3" = "unmount" ]; then
  unmount_ramdisk
else
  create_ramdisk
fi
