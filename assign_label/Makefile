DEBUG_AL = y
AL_SRCS = assign_label.cpp subsynset_checker.cpp
DTOOL_SRCS = subsynsets_checker.cpp

CXX_FLAGS = -std=c++11
AL_OBJS = $(addsuffix .o, $(basename $(AL_SRCS)))
DTOOL_OBJS = $(addsuffix .o, $(basename $(DTOOL_SRCS)))

ifeq ($(DEBUG_AL),y)
	CXX_FLAGS += -DDEBUG_AL
endif

all: assign_label debug_tool

assign_label: $(AL_OBJS)
	g++ $(AL_OBJS) -o assign_label $(CXX_FLAGS)

debug_tool: $(DTOOL_OBJS)
	g++ $(DTOOL_SRCS) -o subsynsets_checker $(CXX_FLAGS)

%.o: %.cpp
	g++ $(CXX_FLAGS) -o $@ -c $^

.PHONY: clean
clean:
	rm -f *.o
	rm -f assign_label
	rm -f subsynsets_checker
