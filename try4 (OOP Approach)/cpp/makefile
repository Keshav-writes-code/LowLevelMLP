CXX = g++
CXXFLAGS = 
TARGET = my_program
SRCDIR = src
SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJDIR = build
OBJS = $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	./$(TARGET)
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJDIR):
	mkdir -p $(OBJDIR)
clean:
	rm -rf $(OBJDIR) $(TARGET)
