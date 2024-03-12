#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1.h"
#include "TH1F.h"
#include "TROOT.h"
#include "TTree.h"
#include <algorithm>
#include <iostream>

void cut() {

  TH1::AddDirectory(kFALSE);

  TFile *f = new TFile("../../20240229_STT.root");
  f->ls();

  TTree *tree = (TTree *)(f->Get("tree"));
  if (!tree) {
    std::cerr << "TTree tree not available in file " << std::endl;
  }

  TCanvas *c_dc0_1_c_raw =
      new TCanvas("c_dc0_1_c_raw", "c_dc0_1_c_raw", 800, 400);
  TCanvas *c_dc1_1_c_raw =
      new TCanvas("c_dc1_1_c_raw", "c_dc1_1_c_raw", 800, 400);
  TCanvas *c_dc2_1_c_raw =
      new TCanvas("c_dc2_1_c_raw", "c_dc2_1_c_raw", 800, 400);
  TCanvas *c_dc3_1_c_raw =
      new TCanvas("c_dc3_1_c_raw", "c_dc3_1_c_raw", 800, 400);
  TCanvas *c_dc4_1_c_raw =
      new TCanvas("c_dc4_1_c_raw", "c_dc4_1_c_raw", 800, 400);
  TCanvas *c_dc5_1_c_raw =
      new TCanvas("c_dc5_1_c_raw", "c_dc5_1_c_raw", 800, 400);
  TCanvas *c_pm6_1_c_raw =
      new TCanvas("c_pm6_1_c_raw", "c_pm6_1_c_raw", 800, 400);
  TCanvas *c_pm7_1_c_raw =
      new TCanvas("c_pm7_1_c_raw", "c_pm7_1_c_raw", 800, 400);

  int const nBinsX = 1000;

  TH1F *dc0_1_c_raw =
      new TH1F("dc0_1_c_raw", "dc0_1_c_raw", nBinsX, -20e-12, 50e-12);
  TH1F *dc1_1_c_raw =
      new TH1F("dc1_1_c_raw", "dc1_1_c_raw", nBinsX, -20e-12, 50e-12);
  TH1F *dc2_1_c_raw =
      new TH1F("dc2_1_c_raw", "dc2_1_c_raw", nBinsX, -20e-12, 50e-12);
  TH1F *dc3_1_c_raw =
      new TH1F("dc3_1_c_raw", "dc3_1_c_raw", nBinsX, -20e-12, 50e-12);
  TH1F *dc4_1_c_raw =
      new TH1F("dc4_1_c_raw", "dc4_1_c_raw", nBinsX, -20e-12, 50e-12);
  TH1F *dc5_1_c_raw =
      new TH1F("dc5_1_c_raw", "dc5_1_c_raw", nBinsX, -20e-12, 50e-12);
  TH1F *pm6_1_c_raw =
      new TH1F("pm6_1_c_raw", "pm6_1_c_raw", nBinsX, -20e-12, 50e-12);
  TH1F *pm7_1_c_raw =
      new TH1F("pm7_1_c_raw", "pm7_1_c_raw", nBinsX, -20e-12, 50e-12);

  TH1F *dc0_1_c_h = new TH1F("dc0_1_c_h", "dc0_1_c_h", nBinsX, -20e-12, 50e-12);
  TH1F *dc1_1_c_h = new TH1F("dc1_1_c_h", "dc1_1_c_h", nBinsX, -20e-12, 50e-12);
  TH1F *dc2_1_c_h = new TH1F("dc2_1_c_h", "dc2_1_c_h", nBinsX, -20e-12, 50e-12);
  TH1F *dc3_1_c_h = new TH1F("dc3_1_c_h", "dc3_1_c_h", nBinsX, -20e-12, 50e-12);
  TH1F *dc4_1_c_h = new TH1F("dc4_1_c_h", "dc4_1_c_h", nBinsX, -20e-12, 50e-12);
  TH1F *dc5_1_c_h = new TH1F("dc5_1_c_h", "dc5_1_c_h", nBinsX, -20e-12, 50e-12);
  TH1F *pm6_1_c_h = new TH1F("pm6_1_c_h", "pm6_1_c_h", nBinsX, -20e-12, 50e-12);
  TH1F *pm7_1_c_h = new TH1F("pm7_1_c_h", "pm7_1_c_h", nBinsX, -20e-12, 50e-12);

  // fit
  TF1 *dc0_1_c_f = new TF1("dc0_1_c_f", "gaus", -3e-12, -2e-12);
  TF1 *dc1_1_c_f = new TF1("dc1_1_c_f", "gaus", -3e-12, -2e-12);
  TF1 *dc2_1_c_f = new TF1("dc2_1_c_f", "gaus", -3e-12, -2e-12);
  TF1 *dc3_1_c_f = new TF1("dc3_1_c_f", "gaus", -3e-12, -2e-12);
  TF1 *dc4_1_c_f = new TF1("dc4_1_c_f", "gaus", -3e-12, -2e-12);
  TF1 *dc5_1_c_f = new TF1("dc5_1_c_f", "gaus", -3e-12, -2e-12);
  TF1 *pm6_1_c_f = new TF1("pm6_1_c_f", "gaus", -3e-12, -2e-12);
  TF1 *pm7_1_c_f = new TF1("pm7_1_c_f", "gaus", -3e-12, -2e-12);

  dc0_1_c_raw->Fit(dc0_1_c_f);
  dc1_1_c_raw->Fit(dc1_1_c_f);
  dc2_1_c_raw->Fit(dc2_1_c_f);
  dc3_1_c_raw->Fit(dc3_1_c_f);
  dc4_1_c_raw->Fit(dc4_1_c_f);
  dc5_1_c_raw->Fit(dc5_1_c_f);
  pm6_1_c_raw->Fit(pm6_1_c_f);
  pm7_1_c_raw->Fit(pm7_1_c_f);

  // drawing histograms
  c_dc0_1_c_raw->cd();
  tree->Draw("dc0_1_c>>dc0_1_c_raw", "", "");
  dc0_1_c_f->Draw("same");

  c_dc1_1_c_raw->cd();
  tree->Draw("dc1_1_c>>dc1_1_c_raw", "", "");
  dc1_1_c_f->Draw("same");

  c_dc2_1_c_raw->cd();
  tree->Draw("dc2_1_c>>dc2_1_c_raw", "", "");
  dc2_1_c_f->Draw("same");

  c_dc3_1_c_raw->cd();
  tree->Draw("dc3_1_c>>dc3_1_c_raw", "", "");
  dc3_1_c_f->Draw("same");

  c_dc4_1_c_raw->cd();
  tree->Draw("dc4_1_c>>dc4_1_c_raw", "", "");
  dc4_1_c_f->Draw("same");

  c_dc5_1_c_raw->cd();
  tree->Draw("dc5_1_c>>dc5_1_c_raw", "", "");
  dc5_1_c_f->Draw("same");

  c_pm6_1_c_raw->cd();
  tree->Draw("pm6_1_c>>pm6_1_c_raw", "", "");
  pm6_1_c_f->Draw("same");

  c_pm7_1_c_raw->cd();
  tree->Draw("pm7_1_c>>pm7_1_c_raw", "", "");
  pm7_1_c_f->Draw("same");

  // printing fit parameters

  std::cout << '\n' << "Fit parameters" << '\n';
  std::cout << "dc0_1_c_f Mean: " << dc1_1_c_f->GetParameter(1) << '\n';
  std::cout << "dc1_1_c_f Mean: " << dc0_1_c_f->GetParameter(1) << '\n';
  std::cout << "dc2_1_c_f Mean: " << dc2_1_c_f->GetParameter(1) << '\n';
  std::cout << "dc3_1_c_f Mean: " << dc3_1_c_f->GetParameter(1) << '\n';
  std::cout << "dc4_1_c_f Mean: " << dc4_1_c_f->GetParameter(1) << '\n';
  std::cout << "dc5_1_c_f Mean: " << dc5_1_c_f->GetParameter(1) << '\n';
  std::cout << "pm6_1_c_f Mean: " << pm6_1_c_f->GetParameter(1) << '\n';
  std::cout << "pm7_1_c_f Mean: " << pm7_1_c_f->GetParameter(1) << '\n';
  std::cout << '\n';

  std::cout << "dc0_1_c_f RMS: " << dc1_1_c_f->GetParameter(2) << '\n';
  std::cout << "dc1_1_c_f RMS: " << dc0_1_c_f->GetParameter(2) << '\n';
  std::cout << "dc2_1_c_f RMS: " << dc2_1_c_f->GetParameter(2) << '\n';
  std::cout << "dc3_1_c_f RMS: " << dc3_1_c_f->GetParameter(2) << '\n';
  std::cout << "dc4_1_c_f RMS: " << dc4_1_c_f->GetParameter(2) << '\n';
  std::cout << "dc5_1_c_f RMS: " << dc5_1_c_f->GetParameter(2) << '\n';
  std::cout << "pm6_1_c_f RMS: " << pm6_1_c_f->GetParameter(2) << '\n';
  std::cout << "pm7_1_c_f RMS: " << pm7_1_c_f->GetParameter(2) << '\n';
  std::cout << '\n';

  int n = 3;
  std::cout << "dc0_1_c_f nRMS: " << dc1_1_c_f->GetParameter(2) * n << '\n';
  std::cout << "dc1_1_c_f nRMS: " << dc0_1_c_f->GetParameter(2) * n << '\n';
  std::cout << "dc2_1_c_f nRMS: " << dc2_1_c_f->GetParameter(2) * n << '\n';
  std::cout << "dc3_1_c_f nRMS: " << dc3_1_c_f->GetParameter(2) * n << '\n';
  std::cout << "dc4_1_c_f nRMS: " << dc4_1_c_f->GetParameter(2) * n << '\n';
  std::cout << "dc5_1_c_f nRMS: " << dc5_1_c_f->GetParameter(2) * n << '\n';
  std::cout << "pm6_1_c_f nRMS: " << pm6_1_c_f->GetParameter(2) * n << '\n';
  std::cout << "pm7_1_c_f nRMS: " << pm7_1_c_f->GetParameter(2) * n << '\n';

  // mean charge collected for each channel

  for (int i = 1; i <= nBinsX; ++i) {
    double cont_dc0_1_c =
        (dc0_1_c_raw->GetBinCenter(i) * dc0_1_c_raw->GetBinContent(i)) /
        (dc0_1_c_raw->GetEntries());
    dc0_1_c_h->SetBinContent(i, cont_dc0_1_c);

    double cont_dc1_1_c =
        (dc1_1_c_raw->GetBinCenter(i) * dc1_1_c_raw->GetBinContent(i)) /
        (dc1_1_c_raw->GetEntries());
    dc1_1_c_h->SetBinContent(i, cont_dc1_1_c);

    double cont_dc2_1_c =
        (dc2_1_c_raw->GetBinCenter(i) * dc2_1_c_raw->GetBinContent(i)) /
        (dc2_1_c_raw->GetEntries());
    dc2_1_c_h->SetBinContent(i, cont_dc2_1_c);

    double cont_dc3_1_c =
        (dc3_1_c_raw->GetBinCenter(i) * dc3_1_c_raw->GetBinContent(i)) /
        (dc3_1_c_raw->GetEntries());
    dc3_1_c_h->SetBinContent(i, cont_dc3_1_c);

    double cont_dc4_1_c =
        (dc4_1_c_raw->GetBinCenter(i) * dc4_1_c_raw->GetBinContent(i)) /
        (dc4_1_c_raw->GetEntries());
    dc4_1_c_h->SetBinContent(i, cont_dc4_1_c);

    double cont_dc5_1_c =
        (dc5_1_c_raw->GetBinCenter(i) * dc5_1_c_raw->GetBinContent(i)) /
        (dc5_1_c_raw->GetEntries());
    dc5_1_c_h->SetBinContent(i, cont_dc5_1_c);

    double cont_pm6_1_c =
        (pm6_1_c_raw->GetBinCenter(i) * pm6_1_c_raw->GetBinContent(i)) /
        (pm6_1_c_raw->GetEntries());
    pm6_1_c_h->SetBinContent(i, cont_pm6_1_c);

    double cont_pm7_1_c =
        (pm7_1_c_raw->GetBinCenter(i) * pm7_1_c_raw->GetBinContent(i)) /
        (pm7_1_c_raw->GetEntries());
    pm7_1_c_h->SetBinContent(i, cont_pm7_1_c);
  }

  double dc0_1_c_mcc =
      dc0_1_c_h->Integral(dc1_1_c_f->GetParameter(2) * n, 50e-12);
  double dc1_1_c_mcc =
      dc1_1_c_h->Integral(dc0_1_c_f->GetParameter(2) * n, 50e-12);
  double dc2_1_c_mcc =
      dc2_1_c_h->Integral(dc2_1_c_f->GetParameter(2) * n, 50e-12);
  double dc3_1_c_mcc =
      dc3_1_c_h->Integral(dc3_1_c_f->GetParameter(2) * n, 50e-12);
  double dc4_1_c_mcc =
      dc4_1_c_h->Integral(dc4_1_c_f->GetParameter(2) * n, 50e-12);
  double dc5_1_c_mcc =
      dc5_1_c_h->Integral(dc5_1_c_f->GetParameter(2) * n, 50e-12);
  double pm6_1_c_mcc =
      pm6_1_c_h->Integral(pm6_1_c_f->GetParameter(2) * n, 50e-12);
  double pm7_1_c_mcc =
      pm7_1_c_h->Integral(pm7_1_c_f->GetParameter(2) * n, 50e-12);

  std::cout << '\n';

  std::cout << "mean charge collected in dc0_1_c:" << dc0_1_c_mcc << '\n';
  std::cout << "mean charge collected in dc1_1_c:" << dc1_1_c_mcc << '\n';
  std::cout << "mean charge collected in dc2_1_c:" << dc2_1_c_mcc << '\n';
  std::cout << "mean charge collected in dc3_1_c:" << dc3_1_c_mcc << '\n';
  std::cout << "mean charge collected in dc4_1_c:" << dc4_1_c_mcc << '\n';
  std::cout << "mean charge collected in dc5_1_c:" << dc5_1_c_mcc << '\n';
  std::cout << "mean charge collected in pm6_1_c:" << pm6_1_c_mcc << '\n';
  std::cout << "mean charge collected in pm7_1_c:" << pm7_1_c_mcc << '\n';

  f->Close();
}