#include <iostream>
#include "convertDigitizer.cpp"

void convert_files(int run_i, int run_f){
    std:string abs_path = "path/to/data/"; //CHANGE

    for (int r = run_i; r <= run_f; r++) {
        std::string binFile1751 = abs_path + "run_" + std::to_string(r) + "_0x17510000v1751.bin";
        std::string rootFile1751 = abs_path + "run_" + std::to_string(r) + "_0x17510000v1751.root";
        std::string binFile1752 = abs_path + "run_" + std::to_string(r) + "_0x17520000v1751.bin";
        std::string rootFile1752 = abs_path + "run_" + std::to_string(r) + "_0x17520000v1751.root";

        std::cout << binFile1751.c_str() << std::endl;
        bin2root_V1751(binFile1751.c_str(), rootFile1751.c_str());

        std::cout << binFile1752.c_str() << std::endl;
        bin2root_V1751(binFile1752.c_str(), rootFile1752.c_str());
    }
}