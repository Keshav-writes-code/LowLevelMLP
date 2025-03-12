#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <sys/ioctl.h> // For terminal size
#include <unistd.h>    // For STDOUT_FILENO
#include "../include/console/progressbar.h"

int getTerminalWidth()
{
  struct winsize w;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  return w.ws_col; // Return the number of columns
}

namespace Console
{
  void showProgressBar(int totalSteps, int currentStep)
  {
    // Dynamically calculate the bar width
    int terminalWidth = getTerminalWidth();
    int barWidth = std::max(terminalWidth - 10, 10); // Ensure a minimum bar width

    // Calculate the percentage completed
    float progress = static_cast<float>(currentStep) / totalSteps;

    // Build the progress bar
    std::string bar = "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i)
    {
      if (i < pos)
        bar += "=";
      else if (i == pos)
        bar += ">";
      else
        bar += " ";
    }
    bar += "] " + std::to_string(int(progress * 100)) + "%";

    // Clear the line and print the new progress bar
    std::cout << "\r" << std::string(terminalWidth, ' ') << "\r"; // Clear line
    std::cout << bar;

    // Flush output for immediate display
    std::cout.flush();
  }
}
