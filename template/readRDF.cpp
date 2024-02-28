#include <iostream>
#include "convertDigitizer.cpp"

void df(int run_number) {
    std:string abs_path = "path/to/data"; //CHANGE
    //ROOT::EnableImplicitMT(); // Tell ROOT you want to go parallel
    ROOT::RDataFrame df("EvFromDig", "run_60_0x17510000v1751.root"); //CHANGE NAME OF run_XX__0x17510000v1751.root
    auto ch0 = df.Histo1D("ch00"); //read channel zero
    ch0->Draw();
}

void convert_files(int run_i, int run_f) {
    std:string abs_path = "path/to/data"; //CHANGE

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

void plot_waveforms(int run_i, int run_f) {
    std:string abs_path = "path/to/data"; //CHANGE

    TChain chain("EvFromDig");
    for (int r = run_i; r <= run_f; r++) {
        std::string run = abs_path + "run_" + std::to_string(r) + "_0x17520000v1751.root";
        chain.Add(run.c_str());
    }
    
    chain.Draw("ch00:Iteration$","Entry$==1200","PL");
    //TODO: store chain in some external varable
}

void plot_tracks(std::string ch, std::string dc_charge, std::string pmt_charge) {
    std:string fname = "path/to/file/filename.root"; //CHANGE

    TFile* T = TFile::Open(fname.c_str());
    TTree* tracks = T->Get<TTree>("tree"); //MODIFY TTREE NAME IF DIFFERENT FORM "tree"

    std::string cond1 = "-pm7_c>" + pmt_charge;
    std::string cond2 = "dc" + ch + "_c>" + dc_charge + "&&" + cond1;

    TCanvas* c1 = new TCanvas("c1", "c1", 0, 0, 700, 600);
    tracks->Draw("y:x>>h1", "", "");
    tracks->Draw("y:x>>h2", cond1.c_str(), "");
    tracks->Draw("y:x>>h3", cond2.c_str(), "");
    TH2F* h1 = (TH2F*)gDirectory->Get("h1");
    TH2F* h2 = (TH2F*)gDirectory->Get("h2");
    TH2F* h3 = (TH2F*)gDirectory->Get("h3");
    h1->SetMarkerColor(kRed);
    h2->SetMarkerColor(kGreen);
    h3->SetMarkerColor(kBlue);
    h3->SetMarkerStyle(7);
    h1->Draw();
    h2->Draw("same");
    h3->Draw("same");

}
