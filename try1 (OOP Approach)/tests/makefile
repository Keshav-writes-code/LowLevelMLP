CXX = g++
CXXFLAGS = 
TARGET = my_program
SRCS = ../main.cpp ../classes.cpp
OBJDIR = build
OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(SRCS)))
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
$(OBJDIR)/%.o: ../%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJDIR):
	mkdir -p $(OBJDIR)
clean:
	rm -rf $(OBJDIR) $(TARGET)
