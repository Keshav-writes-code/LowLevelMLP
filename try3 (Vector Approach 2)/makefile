CXX = g++
CXXFLAGS = 
TARGET = my_program
SRCS = main.cpp classes.cpp
OBJDIR = build
OBJS = $(addprefix $(OBJDIR)/, $(SRCS:.cpp=.o))
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	./$(TARGET)
$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJDIR):
	mkdir -p $(OBJDIR)
run: $(TARGET)
	./$(TARGET)
clean:
	rm -rf $(OBJDIR) $(TARGET)
