.PHONY: all clean


CC = clang
CFLAGS = -O3 -march=armv9-a+sve+sve2+sme -Iinclude 
ASMFLAGS = -O3 -march=armv9-a+sve+sve2+sme -Iinclude 
LINKFLAGS = -rtlib=compiler-rt

DIR_BUILD = build
DIR_EXEC = exec
DIRS = $(DIR_BUILD) $(DIR_EXEC)


EXE = gemm
EXE := $(addprefix $(DIR_EXEC)/, $(EXE))

DIR_SRC = src

C_FILES = $(wildcard $(DIR_SRC)/*.c)
ASM_FILES = $(wildcard $(DIR_SRC)/*.s)
OBJ_FILES = $(C_FILES:$(DIR_SRC)/%.c=$(DIR_BUILD)/%_c.o)
OBJ_FILES += $(ASM_FILES:$(DIR_SRC)/%.s=$(DIR_BUILD)/%_s.o)

DEP_FILES = $(OBJ_FILES:%.o=%.d)



all : $(EXE)

-include $(DEP_FILES)




$(DIR_BUILD)/%_c.o: $(DIR_SRC)/%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -c $< -o $@

$(DIR_BUILD)/%_s.o: $(DIR_SRC)/%.s
	mkdir -p $(@D)
	$(CC) $(ASMFLAGS) -MMD -c $< -o $@


$(EXE): $(OBJ_FILES)
	mkdir -p $(@D)
	$(CC) $(LINKFLAGS) -o $@  $(OBJ_FILES)


run:
	@./$(EXE)

clean:
	rm -rf $(DIRS)

