# Makefile

DOCKER_IMAGE = street-fighter-ai-emulator
VOLUME_MOUNT = $(shell pwd):/home/user
ROOMS_DIR = roms
CLI_PATH = /home/user/src/street_fighter_ai/main.py

import-roms:
	@echo "Importing ROMs from $(ROMS_DIR) using Retro Gym..."
	docker run -it -v $(VOLUME_MOUNT) $(DOCKER_IMAGE) python3 -m retro.import $(ROMS_DIR)
	@echo "ROM import complete!"

.PHONY: train
train:
	docker run -it -v $(VOLUME_MOUNT) $(DOCKER_IMAGE) /bin/bash -c \
		"python3 -m retro.import $(ROMS_DIR) && python $(CLI_PATH) train"

pre-commit:
	docker run -it -v $(VOLUME_MOUNT) $(DOCKER_IMAGE) /bin/bash -c \
		"pre-commit run --all-files"

.PHONY: clean
clean:
	docker system prune -f

.PHONY: launch
launch:
	docker-compose build up --no-cache
