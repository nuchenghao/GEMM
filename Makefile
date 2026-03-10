.PHONY: all clean test


TARGET ?= MAC

CC = clang
LINKFLAGS = -rtlib=compiler-rt -lmatmul -L ./test

ifeq ($(TARGET),MAC)
  MARCH = armv9-a+nosve+sme
  CFLAGS += -DMAC -DACCELERATE_NEW_LAPACK
  LINKFLAGS += -framework Accelerate
  ENTRY = _UnitTest
else ifeq ($(TARGET),LS)
  MARCH = armv9-a+sve+sve2+sme
  CFLAGS += -DLS
  LINKFLAGS += -lkblas -lm
  ENTRY = UnitTest
else
  $(error Unknown TARGET "$(TARGET)". Use MAC or LS)
endif

CFLAGS += -O3 -fno-stack-protector -march=$(MARCH) -Iinclude 
ASMFLAGS = -O3 -fno-stack-protector -march=$(MARCH) -Iinclude

DIR_BUILD = build
DIR_EXEC = exec
DIRS = $(DIR_BUILD) $(DIR_EXEC)


EXE = gemm
EXE := $(addprefix $(DIR_EXEC)/, $(EXE))
EXE_TEST = $(DIR_EXEC)/test
LIB_MATMUL = test/libmatmul.a

VPATH = src:test

C_FILES = $(wildcard src/*.c)
ASM_FILES = $(wildcard src/*.s)
OBJ_FILES = $(addprefix $(DIR_BUILD)/,$(patsubst %.c,%_c.o,$(notdir $(C_FILES))))
OBJ_FILES += $(addprefix $(DIR_BUILD)/,$(patsubst %.s,%_s.o,$(notdir $(ASM_FILES))))

DEP_FILES = $(OBJ_FILES:%.o=%.d) $(patsubst test/%.c,$(DIR_BUILD)/%_c.d,$(wildcard test/*.c))



all : $(EXE)

-include $(DEP_FILES)




$(DIR_BUILD)/%_c.o: %.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -c $< -o $@

$(DIR_BUILD)/%_s.o: %.s
	mkdir -p $(@D)
	$(CC) $(ASMFLAGS) -MMD -c $< -o $@


$(EXE): $(OBJ_FILES)
	mkdir -p $(@D)
	$(CC) $(LINKFLAGS) -o $@  $(OBJ_FILES)

test: $(EXE_TEST)

$(EXE_TEST): $(OBJ_FILES) $(patsubst test/%.c,$(DIR_BUILD)/%_c.o,$(wildcard test/*.c)) $(LIB_MATMUL)
	mkdir -p $(@D)
	$(CC) $(LINKFLAGS) -e $(ENTRY) -o $@ $^

run:
	@./$(EXE)

clean:
	rm -rf $(DIRS)

