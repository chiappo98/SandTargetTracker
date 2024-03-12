#include "TFile.h"
#include "TH2F.h"
#include "TTree.h"
#include <iomanip>
#include <iostream>

const double PI = acos(-1.);

const int nBinsX = 125 * 2;
const int nBinsY = 125 * 2;

// Drift Chamber Layers
TH2F hAllTracks_1("hAllTracks_1", "hAllTracks_1", nBinsX, 0, 50, nBinsY, 0, 50);
TH2F hAllTracks_2("hAllTracks_2", "hAllTracks_2", nBinsX, 0, 50, nBinsY, 0, 50);
TH2F hAllTracks_3("hAllTracks_3", "hAllTracks_3", nBinsX, 0, 50, nBinsY, 0, 50);

// large Scintillator layer->è sul ch6
TH2F hAllTracks_4("hAllTracks_4", "hAllTracks_4", nBinsX, 0, 50, nBinsY, 0, 50);
// small scintillator layer->è sul ch7
TH2F hAllTracks_5("hAllTracks_5", "hAllTracks_5", nBinsX, 0, 50, nBinsY, 0, 50);

// Digitizer V1751 (2)
TH2F hEffDC_D2_ch0("hEffDC_D2_ch0", "hEffDC_D2_ch0", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D2_ch1("hEffDC_D2_ch1", "hEffDC_D2_ch1", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D2_ch2("hEffDC_D2_ch2", "hEffDC_D2_ch2", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D2_ch3("hEffDC_D2_ch3", "hEffDC_D2_ch3", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D2_ch4("hEffDC_D2_ch4", "hEffDC_D2_ch4", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D2_ch5("hEffDC_D2_ch5", "hEffDC_D2_ch5", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffPM_D2_ch7("hEffPM_D2_ch7", "hEffPM_D2_ch7", nBinsX, 0, 50, nBinsY, 0,
                   50);

// Digitizer V1751 (1)
TH2F hEffDC_D1_ch0("hEffDC_D1_ch0", "hEffDC_D1_ch0", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D1_ch1("hEffDC_D1_ch1", "hEffDC_D1_ch1", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D1_ch2("hEffDC_D1_ch2", "hEffDC_D1_ch2", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D1_ch3("hEffDC_D1_ch3", "hEffDC_D1_ch3", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D1_ch4("hEffDC_D1_ch4", "hEffDC_D1_ch4", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffDC_D1_ch5("hEffDC_D1_ch5", "hEffDC_D1_ch5", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffPM_D1_ch6("hEffPM_D1_ch6", "hEffPM_D1_ch6", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffPM_D1_ch7("hEffPM_D1_ch7", "hEffPM_D1_ch7", nBinsX, 0, 50, nBinsY, 0,
                   50);
TH2F hEffPM_D1_ch6_ch7("hEffPM_D1_ch6_ch7", "hEffPM_D1_ch6_ch7", nBinsX, 0, 50, nBinsY,
                       0, 50);

int anaysis(std::string inputFileROOT, std::string outputFileROOT) {
  int entry;

  // tracks info
  int nTracksXZ, nTracksYZ;
  double x, y, z, sx, sy, chi2xz, chi2yz;

  // Drift Chamber Info digitizer V1751 (2)
  double dc0_pp, dc1_pp, dc2_pp, dc3_pp, dc4_pp,
      dc5_pp; // pulse amplitude (pp peak peak)
  double dc0_c, dc1_c, dc2_c, dc3_c, dc4_c, dc5_c; // pulse integral
  double dc0_t10, dc1_t10, dc2_t10, dc3_t10, dc4_t10,
      dc5_t10;                   // pulse time at 10% amplitude
  double pm7_pp, pm7_c, pm7_t10; // PMT info (small scintillator)

  // Drift Chamber Info digitizer V1751 (1)
  double dc0_1_pp, dc1_1_pp, dc2_1_pp, dc3_1_pp, dc4_1_pp,
      dc5_1_pp; // pulse amplitude
  double dc0_1_c, dc1_1_c, dc2_1_c, dc3_1_c, dc4_1_c, dc5_1_c; // pulse integral
  double dc0_1_t10, dc1_1_t10, dc2_1_t10, dc3_1_t10, dc4_1_t10,
      dc5_1_t10;                       // pulse time at 10% amplitude
  double pm6_1_pp, pm6_1_c, pm6_1_t10; // PMT info (large scintillator)
  double pm7_1_pp, pm7_1_c, pm7_1_t10; // PMT info (large scintillator)

  TFile f(inputFileROOT.c_str());
  if (!(f.IsOpen())) {
    std::cerr << "file " << inputFileROOT << " not found" << std::endl;
    return -1;
  }

  TTree *tree = (TTree *)(f.Get("tree"));
  if (!tree) {
    std::cerr << "TTree tree not available in file " << inputFileROOT
              << std::endl;
    return -1;
  }

  tree->SetBranchAddress("entry", &entry);
  // track info
  tree->SetBranchAddress("x", &x);
  tree->SetBranchAddress("y", &y);
  tree->SetBranchAddress("z", &z);
  tree->SetBranchAddress("sx", &sx);
  tree->SetBranchAddress("sy", &sy);
  tree->SetBranchAddress("nTracksXZ", &nTracksXZ);
  tree->SetBranchAddress("nTracksYZ", &nTracksYZ);
  tree->SetBranchAddress("chi2xz", &chi2xz);
  tree->SetBranchAddress("chi2yz", &chi2yz);

  // Drift Chamber V1751 (2)
  tree->SetBranchAddress("dc0_pp", &dc0_pp);
  tree->SetBranchAddress("dc1_pp", &dc1_pp);
  tree->SetBranchAddress("dc2_pp", &dc2_pp);
  tree->SetBranchAddress("dc3_pp", &dc3_pp);
  tree->SetBranchAddress("dc4_pp", &dc4_pp);
  tree->SetBranchAddress("dc5_pp", &dc5_pp);
  tree->SetBranchAddress("pm7_pp", &pm7_pp);

  tree->SetBranchAddress("dc0_c", &dc0_c);
  tree->SetBranchAddress("dc1_c", &dc1_c);
  tree->SetBranchAddress("dc2_c", &dc2_c);
  tree->SetBranchAddress("dc3_c", &dc3_c);
  tree->SetBranchAddress("dc4_c", &dc4_c);
  tree->SetBranchAddress("dc5_c", &dc5_c);
  tree->SetBranchAddress("pm7_c", &pm7_c);

  tree->SetBranchAddress("dc0_t", &dc0_t10);
  tree->SetBranchAddress("dc1_t", &dc1_t10);
  tree->SetBranchAddress("dc2_t", &dc2_t10);
  tree->SetBranchAddress("dc3_t", &dc3_t10);
  tree->SetBranchAddress("dc4_t", &dc4_t10);
  tree->SetBranchAddress("dc5_t", &dc5_t10);
  tree->SetBranchAddress("pm7_t", &pm7_t10);

  // Drift Chamber V1751 (1)
  tree->SetBranchAddress("dc0_1_pp", &dc0_1_pp);
  tree->SetBranchAddress("dc1_1_pp", &dc1_1_pp);
  tree->SetBranchAddress("dc2_1_pp", &dc2_1_pp);
  tree->SetBranchAddress("dc3_1_pp", &dc3_1_pp);
  tree->SetBranchAddress("dc4_1_pp", &dc4_1_pp);
  tree->SetBranchAddress("dc5_1_pp", &dc5_1_pp);
  tree->SetBranchAddress("pm6_1_pp", &pm6_1_pp);
  tree->SetBranchAddress("pm7_1_pp", &pm7_1_pp);

  tree->SetBranchAddress("dc0_1_c", &dc0_1_c);
  tree->SetBranchAddress("dc1_1_c", &dc1_1_c);
  tree->SetBranchAddress("dc2_1_c", &dc2_1_c);
  tree->SetBranchAddress("dc3_1_c", &dc3_1_c);
  tree->SetBranchAddress("dc4_1_c", &dc4_1_c);
  tree->SetBranchAddress("dc5_1_c", &dc5_1_c);
  tree->SetBranchAddress("pm6_1_c", &pm6_1_c);
  tree->SetBranchAddress("pm7_1_c", &pm7_1_c);

  tree->SetBranchAddress("dc0_1_t", &dc0_1_t10);
  tree->SetBranchAddress("dc1_1_t", &dc1_1_t10);
  tree->SetBranchAddress("dc2_1_t", &dc2_1_t10);
  tree->SetBranchAddress("dc3_1_t", &dc3_1_t10);
  tree->SetBranchAddress("dc4_1_t", &dc4_1_t10);
  tree->SetBranchAddress("dc5_1_t", &dc5_1_t10);
  tree->SetBranchAddress("pm6_1_t", &pm6_1_t10);
  tree->SetBranchAddress("pm7_1_t", &pm7_1_t10);

  // DZ to project tracks
  double dzDC_1 = 20.3 + 0.2 + 0.5;
  double dzDC_2 = dzDC_1 + 1;
  double dzDC_3 = dzDC_2 + 1;
  double dzScintLarge = dzDC_3 + 0.5 + 2;
  double dzScint = dzDC_3 + 0.5 + 2 + 1 + 0.8;

  std::cout << "dz DC plane 1: " << dzDC_1 << std::endl;
  std::cout << "dz DC plane 2: " << dzDC_2 << std::endl;
  std::cout << "dz DC plane 3: " << dzDC_3 << std::endl;
  std::cout << "dz scint Scint Large: " << dzScintLarge << std::endl;
  std::cout << "dz scint LIMADOU: " << dzScint << std::endl;

  int nEvents = tree->GetEntries(); // get the total number of events

  std::cout << "Processing " << nEvents << " events: " << std::setw(2) << 0
            << "%";

  int nSel = 0;

  for (int iEv = 0; iEv < nEvents; iEv++) {
    std::cout << "\b\b\b" << std::setprecision(2) << std::setw(2)
              << int(double(iEv) / double(nEvents - 1) * 100) << "%"
              << std::flush;

    tree->GetEntry(iEv); // get the eventand fill the variables

    double den = sqrt(1 + sx * sx + sy * sy);
    double angle = acos(1. / den);

    if (nTracksXZ != 1 || nTracksYZ != 1)
      continue;

    nSel++;

    double x1 = x + dzDC_1 * sx;
    double y1 = y + dzDC_1 * sy;

    double x2 = x + dzDC_2 * sx;
    double y2 = y + dzDC_2 * sy;

    double x3 = x + dzDC_3 * sx;
    double y3 = y + dzDC_3 * sy;

    double xScintLarge = x + dzScintLarge * sx;
    double yScintLarge = y + dzScintLarge * sy;

    double xScint = x + dzScint * sx;
    double yScint = y + dzScint * sy;

    hAllTracks_1.Fill(x1, y1);
    hAllTracks_2.Fill(x2, y2);
    hAllTracks_3.Fill(x3, y3);
    hAllTracks_4.Fill(xScintLarge, yScintLarge);
    hAllTracks_5.Fill(xScint, yScint);

    // Drift V1751 (2)
    if (dc0_c > 7.0e-12) {
      int bin = hEffDC_D2_ch0.Fill(x3, y3); // fill the efficiency
      // int bin -- bin in cui cade il punto x3, y3
    }
    if (dc1_c > 7.0e-12) {
      int bin = hEffDC_D2_ch1.Fill(x2, y2); // fill the efficiency
    }
    if (dc2_c > 12.0e-12) {
      int bin = hEffDC_D2_ch2.Fill(x1, y1); // fill the efficiency
    }
    if (dc3_c > 10.0e-12) {
      int bin = hEffDC_D2_ch3.Fill(x3, y3); // fill the efficiency
    }
    if (dc4_c > 15.0e-12) {
      int bin = hEffDC_D2_ch4.Fill(x2, y2); // fill the efficiency
    }
    if (dc5_c > 20.0e-12) {
      int bin = hEffDC_D2_ch5.Fill(x1, y1); // fill the efficiency
    }
    if (pm7_c > 4e-12) {
      int bin = hEffPM_D2_ch7.Fill(xScint, yScint); // fill the efficiency
    }

    // Drift V1751 (1)
    if (dc0_1_c > 7.0e-12) {
      int bin = hEffDC_D1_ch0.Fill(x3, y3);
    }
    if (dc1_1_c > 7.0e-12) {
      int bin = hEffDC_D1_ch1.Fill(x2, y2);
    }
    if (dc2_1_c > 12.0e-12) {
      int bin = hEffDC_D1_ch2.Fill(x1, y1);
    }
    if (dc3_1_c > 10.0e-12) {
      int bin = hEffDC_D1_ch3.Fill(x3, y3);
    }
    if (dc4_1_c > 15.0e-12) {
      int bin = hEffDC_D1_ch4.Fill(x2, y2);
    }
    if (dc5_1_c > 20.0e-12) {
      int bin = hEffDC_D1_ch5.Fill(x1, y1);
    }
    if (pm6_1_c > 4.0e-12) {
      int bin = hEffPM_D1_ch6.Fill(xScintLarge, yScintLarge);
    }
    if (pm7_1_c > 4.0e-12) {
      int bin = hEffPM_D1_ch7.Fill(xScintLarge, yScintLarge);
    }
    if (pm6_1_c > 4.0e-12 || pm7_1_c > 4.0e-12) {
      int bin = hEffPM_D1_ch6_ch7.Fill(xScintLarge, yScintLarge);
    }
  }
  std::cout << std::endl;
  std::cout << "nSelected: " << nSel << std::endl;

  for (int i = 1; i <= nBinsX * nBinsY; i++) { // perchè sono in 2D
    double den1 = hAllTracks_1.GetBinContent(i);
    double den2 = hAllTracks_2.GetBinContent(i);
    double den3 = hAllTracks_3.GetBinContent(i);
    double den4 = hAllTracks_4.GetBinContent(i);
    double den5 = hAllTracks_5.GetBinContent(i);

    // efficienza sul canale D2
    double numEffDC_D2_ch0 = hEffDC_D2_ch0.GetBinContent(i);
    double numEffDC_D2_ch1 = hEffDC_D2_ch1.GetBinContent(i);
    double numEffDC_D2_ch2 = hEffDC_D2_ch2.GetBinContent(i);
    double numEffDC_D2_ch3 = hEffDC_D2_ch3.GetBinContent(i);
    double numEffDC_D2_ch4 = hEffDC_D2_ch4.GetBinContent(i);
    double numEffDC_D2_ch5 = hEffDC_D2_ch5.GetBinContent(i);
    double numEffPM_D2_ch7 = hEffPM_D2_ch7.GetBinContent(i);

    if (den3 != 0) {
      hEffDC_D2_ch0.SetBinContent(i, numEffDC_D2_ch0 / den3);
      hEffDC_D2_ch3.SetBinContent(i, numEffDC_D2_ch3 / den3);
    }
    if (den2 != 0) {
      hEffDC_D2_ch1.SetBinContent(i, numEffDC_D2_ch1 / den2);
      hEffDC_D2_ch4.SetBinContent(i, numEffDC_D2_ch4 / den2);
    }
    if (den1 != 0) {
      hEffDC_D2_ch2.SetBinContent(i, numEffDC_D2_ch2 / den1);
      hEffDC_D2_ch5.SetBinContent(i, numEffDC_D2_ch5 / den1);
    }
    if (den5 != 0)
      hEffPM_D2_ch7.SetBinContent(i, numEffPM_D2_ch7 / den5);

    // efficienza sul canale D1
    double numEffDC_D1_ch0 = hEffDC_D1_ch0.GetBinContent(i);
    double numEffDC_D1_ch1 = hEffDC_D1_ch1.GetBinContent(i);
    double numEffDC_D1_ch2 = hEffDC_D1_ch2.GetBinContent(i);
    double numEffDC_D1_ch3 = hEffDC_D1_ch3.GetBinContent(i);
    double numEffDC_D1_ch4 = hEffDC_D1_ch4.GetBinContent(i);
    double numEffDC_D1_ch5 = hEffDC_D1_ch5.GetBinContent(i);
    double numEffPM_D1_ch6 = hEffPM_D1_ch6.GetBinContent(i);
    double numEffPM_D1_ch7 = hEffPM_D1_ch7.GetBinContent(i);
    double numEffPM_D1_ch6_ch7 = hEffPM_D1_ch6_ch7.GetBinContent(i);

    if (den3 != 0) {
      hEffDC_D1_ch0.SetBinContent(i, numEffDC_D1_ch0 / den3);
      hEffDC_D1_ch3.SetBinContent(i, numEffDC_D1_ch3 / den3);
    }
    if (den2 != 0) {
      hEffDC_D1_ch1.SetBinContent(i, numEffDC_D1_ch1 / den2);
      hEffDC_D1_ch4.SetBinContent(i, numEffDC_D1_ch4 / den2);
    }
    if (den1 != 0) {
      hEffDC_D1_ch2.SetBinContent(i, numEffDC_D1_ch2 / den1);
      hEffDC_D1_ch5.SetBinContent(i, numEffDC_D1_ch5 / den1);
    }
    if (den4 != 0) {
      hEffPM_D1_ch6.SetBinContent(i, numEffPM_D1_ch6 / den4);
      hEffPM_D1_ch7.SetBinContent(i, numEffPM_D1_ch7 / den4);
      hEffPM_D1_ch6_ch7.SetBinContent(i, numEffPM_D1_ch6_ch7 / den4);
    }
  }

  TFile g(outputFileROOT.c_str(), "RECREATE");

  hEffDC_D2_ch0.Write();
  hEffDC_D2_ch1.Write();
  hEffDC_D2_ch2.Write();
  hEffDC_D2_ch3.Write();
  hEffDC_D2_ch4.Write();
  hEffDC_D2_ch5.Write();
  hEffPM_D2_ch7.Write();

  hEffDC_D1_ch0.Write();
  hEffDC_D1_ch1.Write();
  hEffDC_D1_ch2.Write();
  hEffDC_D1_ch3.Write();
  hEffDC_D1_ch4.Write();
  hEffDC_D1_ch5.Write();
  hEffPM_D1_ch6.Write();
  hEffPM_D1_ch7.Write();
  hEffPM_D1_ch6_ch7.Write();
  g.Close();
  f.Close();
  return 0;
}
