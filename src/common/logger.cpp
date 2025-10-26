#include "logger.h"

#include <string.h>
#include <iomanip>
#include <stddef.h>
#include <cstdint>
#include <fstream>

#include <omp.h>

using namespace std;

struct Logger
{
    double time_init;
    ofstream log_file;
};

static Logger logger;


void LogInit(int argc, char **argv)
{
    string log_file_name = "log.log", log_dir = ".";

    for (int i = 1; i < argc; i++)
    {
        auto option = string(argv[i]);

        if (option == "-logfile")
        {
            log_file_name.assign(string(argv[i + 1]));
            i++;
        }
        else if (option == "-logdir")
        {
            log_dir.assign(string(argv[i + 1]));
            i++;
        }
    }

    if (!log_file_name.empty())
    {
        string full_file_path = "";

        if (!log_dir.empty())
        {
            full_file_path.append(log_dir);
            full_file_path.push_back('/');
        }

        full_file_path.append(log_file_name);

        logger.log_file.open(full_file_path, ios_base::out | ios_base::trunc);
        logger.time_init = omp_get_wtime();
    }
}


void LogWrite(const string &s)
{
    if (logger.log_file.is_open())
    {
        auto timestamp = omp_get_wtime() - logger.time_init;
        logger.log_file << '[' << fixed << setprecision(5) << setw(13) << setfill('0') << timestamp << ']' << ' ' << s << endl;
    }
}


void LogFlush()
{
    if (logger.log_file.is_open())
        logger.log_file.flush();
}


void LogFinalize()
{
    logger.log_file.close();
}