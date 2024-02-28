#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

const int SIZE_OF_UNSIGNED_INT = 4;
const int SIZE_OF_UNSIGNED_SHORT = 2;
const int MAX_SIZE = 65536;
const double SCALE_V1720 = 2. / 4096;
const double SCALE_V1751 = 1. / 1024;  //

int bin2root_V1720(const char *inputBinFile,
                   const char *outputRootFile = "tmp.root") {
  std::cout << "Converting digitizer V1720..." << std::endl;
  int startTime = clock();
  UInt_t id, timeStamp, nSamples;
  UShort_t channelConfiguration;
  Double_t ch00[MAX_SIZE];
  Double_t ch01[MAX_SIZE];
  Double_t ch02[MAX_SIZE];
  Double_t ch03[MAX_SIZE];
  Double_t ch04[MAX_SIZE];
  Double_t ch05[MAX_SIZE];
  Double_t ch06[MAX_SIZE];
  Double_t ch07[MAX_SIZE];
  std::cout << "SCALE: " << SCALE_V1720 << std::endl;
  TFile outFile(outputRootFile, "RECREATE");
  TTree *tree = new TTree("EventsFromDigitizer", "EventsFromDigitizer");
  tree->Branch("id", &id, "id/i");  // event id
  tree->Branch("tStamp", &timeStamp,
               "timeStamp/i");  // time stamp of the event
  tree->Branch("chConfig", &channelConfiguration,
               "chConfig/s");  // channel configuration
  tree->Branch("nSamples", &nSamples,
               "nSamples/i");                      // nSamples (each channel)
  tree->Branch("ch00", ch00, "ch00[nSamples]/D");  // Channels [0-7]
  tree->Branch("ch01", ch01, "ch01[nSamples]/D");
  tree->Branch("ch02", ch02, "ch02[nSamples]/D");
  tree->Branch("ch03", ch03, "ch03[nSamples]/D");
  tree->Branch("ch04", ch04, "ch04[nSamples]/D");
  tree->Branch("ch05", ch05, "ch05[nSamples]/D");
  tree->Branch("ch06", ch06, "ch06[nSamples]/D");
  tree->Branch("ch07", ch07, "ch07[nSamples]/D");

  std::ifstream in(inputBinFile, std::ios::binary);
  unsigned int readD32;
  int counter = 0;
  while (!in.eof() && in.good())  // don't know how many events are in the file
  {
    // empty the channels values
    for (int j = 0; j < MAX_SIZE; j++) {
      ch00[j] = 0;
      ch01[j] = 0;
      ch02[j] = 0;
      ch03[j] = 0;
      ch04[j] = 0;
      ch05[j] = 0;
      ch06[j] = 0;
      ch07[j] = 0;
    }
    // reading event headers
    unsigned int byte = in.tellg();

    in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);
    if (in.eof())  // eof reached
      break;
    if (!(readD32 & 0xa0000000)) {
      std::cout << "Found wrong header...process aborted" << std::endl;
      tree->Write();
      outFile.Close();
      in.close();
      return -1;
    }
    counter++;
    int eventSize = (readD32 & 0xFFFFFFF);
    in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);
    channelConfiguration = (readD32 & 0xFF);
    in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);
    id = (readD32 & 0x7FFFFF);
    in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);
    timeStamp = readD32;
    int nChannels = 0;
    std::vector<short> chId;
    for (int j = 0; j < 8; j++) {
      if ((channelConfiguration >> j) & 0x1) {
        chId.push_back(j);
        nChannels++;
      }
    }
    eventSize -= 4;  // size decreased by the head amount.
    int nWordsPerChannel =
        eventSize / nChannels;  // actually this is of D16 words which are
                                // nSamples (D32 words are half of this);
    nSamples = nWordsPerChannel * 2;

    for (int ich = 0; ich < nChannels; ich++) {
      for (int iw = 0; iw < nWordsPerChannel; iw++) {
        in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);
        unsigned short word1 = readD32 & 0x0000FFFF;
        unsigned short word2 = ((readD32 & 0xFFFF0000) >> 16);
        switch (chId.at(ich)) {
          case 0:
            ch00[2 * iw] = word1 * SCALE_V1720;
            ch00[2 * iw + 1] = word2 * SCALE_V1720;
            break;
          case 1:
            ch01[2 * iw] = word1 * SCALE_V1720;
            ch01[2 * iw + 1] = word2 * SCALE_V1720;
            break;
          case 2:
            ch02[2 * iw] = word1 * SCALE_V1720;
            ch02[2 * iw + 1] = word2 * SCALE_V1720;
            break;
          case 3:
            ch03[2 * iw] = word1 * SCALE_V1720;
            ch03[2 * iw + 1] = word2 * SCALE_V1720;
            break;
          case 4:
            ch04[2 * iw] = word1 * SCALE_V1720;
            ch04[2 * iw + 1] = word2 * SCALE_V1720;
            break;
          case 5:
            ch05[2 * iw] = word1 * SCALE_V1720;
            ch05[2 * iw + 1] = word2 * SCALE_V1720;
            break;
          case 6:
            ch06[2 * iw] = word1 * SCALE_V1720;
            ch06[2 * iw + 1] = word2 * SCALE_V1720;
            break;
          case 7:
            ch07[2 * iw] = word1 * SCALE_V1720;
            ch07[2 * iw + 1] = word2 * SCALE_V1720;
            break;
        }
      }
    }

    tree->Fill();
  }
  tree->Write("", TObject::kOverwrite);
  outFile.Close();

  in.close();
  int stopTime = clock();
  std::cout << "Read " << counter << " events in "
            << (stopTime - startTime) / 1000 << " s" << std::endl;
  return 0;
}

int bin2root_V1751(const char *inputBinFile,
                   const char *outputRootFile = "tmp.root") {
  std::cout << "Converting digitizer V1751..." << std::endl;
  int startTime = clock();
  UInt_t id, timeStamp, nSamples;
  UShort_t channelConfiguration;
  Double_t ch00[MAX_SIZE];
  Double_t ch01[MAX_SIZE];
  Double_t ch02[MAX_SIZE];
  Double_t ch03[MAX_SIZE];
  Double_t ch04[MAX_SIZE];
  Double_t ch05[MAX_SIZE];
  Double_t ch06[MAX_SIZE];
  Double_t ch07[MAX_SIZE];
  std::cout << "SCALE: " << SCALE_V1751 << std::endl;
  TFile outFile(outputRootFile, "RECREATE");
  TTree *tree = new TTree("EventsFromDigitizer", "EventsFromDigitizer");
  tree->Branch("id", &id, "id/i");  // event id
  tree->Branch("tStamp", &timeStamp,
               "timeStamp/i");  // time stamp of the event
  tree->Branch("chConfig", &channelConfiguration,
               "chConfig/s");  // channel configuration
  tree->Branch("nSamples", &nSamples,
               "nSamples/i");                      // nSamples (each channel)
  tree->Branch("ch00", ch00, "ch00[nSamples]/D");  // Channels [0-7]
  tree->Branch("ch01", ch01, "ch01[nSamples]/D");
  tree->Branch("ch02", ch02, "ch02[nSamples]/D");
  tree->Branch("ch03", ch03, "ch03[nSamples]/D");
  tree->Branch("ch04", ch04, "ch04[nSamples]/D");
  tree->Branch("ch05", ch05, "ch05[nSamples]/D");
  tree->Branch("ch06", ch06, "ch06[nSamples]/D");
  tree->Branch("ch07", ch07, "ch07[nSamples]/D");

  std::ifstream in(inputBinFile, std::ios::binary);
  unsigned int readD32;
  int counter = 0;
  while (!in.eof() && in.good())  // don't know how many events are in the file
  {
    // empty the channels values
    for (int j = 0; j < MAX_SIZE; j++) {
      ch00[j] = 0;
      ch01[j] = 0;
      ch02[j] = 0;
      ch03[j] = 0;
      ch04[j] = 0;
      ch05[j] = 0;
      ch06[j] = 0;
      ch07[j] = 0;
    }
    // reading event headers
    unsigned int byte = in.tellg();
    in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);

    if (in.eof())  // eof reached
      break;
    if (!(readD32 & 0xa0000000)) {
      std::cout << "Found wrong header...process aborted" << std::endl;
      tree->Write();
      outFile.Close();
      in.close();
      return -1;
    }

    counter++;
    int eventSize = (readD32 & 0xFFFFFFF);
    in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);
    channelConfiguration = (readD32 & 0xFF);
    in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);
    id = (readD32 & 0x7FFFFF);
    in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);
    timeStamp = readD32;
    int nChannels = 0;

    std::vector<short> chId;
    for (int j = 0; j < 8; j++) {
      if ((channelConfiguration >> j) & 0x1) {
        chId.push_back(j);
        nChannels++;
      }
    }
    eventSize -= 4;  // size decreased by the head amount.

    int nWordsPerChannel =
        eventSize / nChannels;  // actually this is of D16 words which are
                                // nSamples (D32 words are half of this);

    // nSamples = nWordsPerChannel * 2;
    for (int ich = 0; ich < nChannels; ich++) {
      int index = 0;
      // char a; std::cin >> a;
      for (int iw = 0; iw < nWordsPerChannel; iw++) {
        in.read(reinterpret_cast<char *>(&readD32), SIZE_OF_UNSIGNED_INT);
        // std::cout << "readD32: " << readD32 << std::endl;
        int nWords = (readD32 >> 30);
        unsigned short word1 = readD32 & 0x3FF;
        unsigned short word2 = (readD32 >> 10) & 0x3FF;
        unsigned short word3 = (readD32 >> 20) & 0x3FF;

        switch (chId.at(ich)) {
          case 0: {
            switch (nWords) {
              case 1: {
                ch00[index++] = word1 * SCALE_V1751;
              } break;
              case 2: {
                ch00[index++] = word1 * SCALE_V1751;
                ch00[index++] = word2 * SCALE_V1751;
              } break;
              case 3: {
                ch00[index++] = word1 * SCALE_V1751;
                ch00[index++] = word2 * SCALE_V1751;
                ch00[index++] = word3 * SCALE_V1751;
              } break;
            }

          } break;

          case 1: {
            switch (nWords) {
              case 1: {
                ch01[index++] = word1 * SCALE_V1751;
              } break;
              case 2: {
                ch01[index++] = word1 * SCALE_V1751;
                ch01[index++] = word2 * SCALE_V1751;
              } break;
              case 3: {
                ch01[index++] = word1 * SCALE_V1751;
                ch01[index++] = word2 * SCALE_V1751;
                ch01[index++] = word3 * SCALE_V1751;
              } break;
            }

          } break;

          case 2: {
            switch (nWords) {
              case 1: {
                ch02[index++] = word1 * SCALE_V1751;
              } break;
              case 2: {
                ch02[index++] = word1 * SCALE_V1751;
                ch02[index++] = word2 * SCALE_V1751;
              } break;
              case 3: {
                ch02[index++] = word1 * SCALE_V1751;
                ch02[index++] = word2 * SCALE_V1751;
                ch02[index++] = word3 * SCALE_V1751;
              } break;
            }

          } break;
          case 3: {
            switch (nWords) {
              case 1: {
                ch03[index++] = word1 * SCALE_V1751;
              } break;
              case 2: {
                ch03[index++] = word1 * SCALE_V1751;
                ch03[index++] = word2 * SCALE_V1751;
              } break;
              case 3: {
                ch03[index++] = word1 * SCALE_V1751;
                ch03[index++] = word2 * SCALE_V1751;
                ch03[index++] = word3 * SCALE_V1751;
              } break;
            }

          } break;
          case 4: {
            switch (nWords) {
              case 1: {
                ch04[index++] = word1 * SCALE_V1751;
              } break;
              case 2: {
                ch04[index++] = word1 * SCALE_V1751;
                ch04[index++] = word2 * SCALE_V1751;
              } break;
              case 3: {
                ch04[index++] = word1 * SCALE_V1751;
                ch04[index++] = word2 * SCALE_V1751;
                ch04[index++] = word3 * SCALE_V1751;
              } break;
            }

          } break;
          case 5: {
            switch (nWords) {
              case 1: {
                ch05[index++] = word1 * SCALE_V1751;
              } break;
              case 2: {
                ch05[index++] = word1 * SCALE_V1751;
                ch05[index++] = word2 * SCALE_V1751;
              } break;
              case 3: {
                ch05[index++] = word1 * SCALE_V1751;
                ch05[index++] = word2 * SCALE_V1751;
                ch05[index++] = word3 * SCALE_V1751;
              } break;
            }

          } break;
          case 6: {
            switch (nWords) {
              case 1: {
                ch06[index++] = word1 * SCALE_V1751;
              } break;
              case 2: {
                ch06[index++] = word1 * SCALE_V1751;
                ch06[index++] = word2 * SCALE_V1751;
              } break;
              case 3: {
                ch06[index++] = word1 * SCALE_V1751;
                ch06[index++] = word2 * SCALE_V1751;
                ch06[index++] = word3 * SCALE_V1751;
              } break;
            }

          } break;
          case 7: {
            switch (nWords) {
              case 1: {
                ch07[index++] = word1 * SCALE_V1751;
              } break;
              case 2: {
                ch07[index++] = word1 * SCALE_V1751;
                ch07[index++] = word2 * SCALE_V1751;
              } break;
              case 3: {
                ch07[index++] = word1 * SCALE_V1751;
                ch07[index++] = word2 * SCALE_V1751;
                ch07[index++] = word3 * SCALE_V1751;
              } break;
            }

          } break;
        }
      }
      nSamples = index;
    }

    tree->Fill();
  }
  tree->Write("", TObject::kOverwrite);
  outFile.Close();

  in.close();
  int stopTime = clock();
  std::cout << "Read " << counter << " events in "
            << (stopTime - startTime) / 1000 << " s" << std::endl;
  return 0;
}
