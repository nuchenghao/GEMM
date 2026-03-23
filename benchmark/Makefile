.PHONY: all clean

CC = clang
CXX = clang++
TARGET ?= MAC
ifeq ($(TARGET),MAC)
  MARCH = armv9-a+nosve+sme
else ifeq ($(TARGET),HW)
  MARCH = armv9-a+sve+sve2+sme
else
  $(error Unknown TARGET "$(TARGET)". Use MAC or HW)
endif


CFLAGS += -O3 -fno-stack-protector -march=$(MARCH) # C compiler options
CXXFLAGS += -O3 -fno-stack-protector -march=$(MARCH) # C++ compiler options
ASMFLAGS += -O3 -fno-stack-protector -march=$(MARCH)

# store object files and dependency files. Dont have extra blankspace " "
DIR_BUILD = build
# store executable files. Dont have extra blankspace " "
DIR_EXEC = exec
DIRS = $(DIR_BUILD) $(DIR_EXEC) # store all directories. When clean, just delete them.


EXE = complicated 
EXE := $(addprefix $(DIR_EXEC)/, $(EXE)) # executable file


# source files
C_FILES   = $(wildcard *.c)
CPP_FILES = $(wildcard *.cpp)
ASM_FILES = $(wildcard *.S)
# object files
OBJS  = $(C_FILES:%.c=$(DIR_BUILD)/%_c.o)
OBJS += $(CPP_FILES:%.cpp=$(DIR_BUILD)/%_cpp.o)
OBJS += $(ASM_FILES:%.S=$(DIR_BUILD)/%_S.o)
# dependency files
DEPS = $(OBJS:.o=.d)




all: $(EXE)


-include $(DEPS)


$(EXE): $(OBJS)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(DIR_BUILD)/%_c.o: %.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -c $< -o $@ 

$(DIR_BUILD)/%_cpp.o: %.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@ 

$(DIR_BUILD)/%_S.o: %.S
	mkdir -p $(@D)
	$(CC) $(ASMFLAGS) -MMD -c $< -o $@ 

run:
	@./$(EXE)

clean:
	rm -rf $(DIRS) 