#pragma once

#include <string>

void LogInit(int argc, char **argv);
void LogWrite(const std::string &s);
void LogFlush();
void LogFinalize();