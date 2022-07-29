HYPOTESTDIR := data/hypotest/
TESTERRORDIR := data/hypotest-error/

hypotest_dirs = $(wildcard $(HYPOTESTDIR)*/)
hypotest_errors := $(hypotest_dirs:$(HYPOTESTDIR)%/=$(TESTERRORDIR)%.csv)

debug:
	@echo $(hypotest_dirs)
	@echo $(hypotest_errors)

all: $(hypotest_errors)

$(hypotest_errors): $(TESTERRORDIR)%.csv: $(HYPOTESTDIR)%/
	hypotest-error --name hypotest-error/$(basename $(notdir $@)) --result_dir=$<

