ROOT := $(CURDIR)
SUBMODULE_DIRS := $(shell git config --file .gitmodules --get-regexp path | awk '{print $$2}')
GIT ?= git

.PHONY: init-submodules pull-submodules push-submodules update-submodules reset-submodules status

init-submodules:
	$(GIT) submodule update --init --recursive

pull-submodules:
	@for dir in $(SUBMODULE_DIRS); do \
		echo ">>> Pulling submodule $$dir"; \
		cd $$dir && $(GIT) pull origin main || true; \
	done

update-submodules:
	@for dir in $(SUBMODULE_DIRS); do \
		echo ">>> Committing and pushing $$dir"; \
		cd $$dir && \
		$(GIT) add . && \
		$(GIT) commit -m "Update submodule $$dir" || true; \
		$(GIT) push origin main || true; \
	done

push-submodules:
	@for dir in $(SUBMODULE_DIRS); do \
		echo ">>> Pushing $$dir"; \
		cd $$dir && $(GIT) push origin main || true; \
	done

reset-submodules:
	@for dir in $(SUBMODULE_DIRS); do \
		echo ">>> Resetting $$dir"; \
		cd $$dir && $(GIT) fetch origin && $(GIT) reset --hard origin/main; \
	done

status:
	@echo "==== MAIN REPO ===="
	$(GIT) status
	@echo "==== SUBMODULES ===="
	@for dir in $(SUBMODULE_DIRS); do \
		echo "--- $$dir ---"; \
		cd $$dir && $(GIT) status; \
	done
