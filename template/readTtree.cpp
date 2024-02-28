#include <iostream>

TTree* read_data(const char* fname) {
	
	TFile* T = TFile::Open(fname); //fname IS THE NAME OF THE ROOT FILE, assumed to be in the same folder of this file
	TTree* tree = T->Get<TTree>("EventsFromDigitizer"); //MODIFY IF TTREE NAME IS DIFFERENT FROM "EventsFromDigitizer"

	return tree;
}

void plot_CHvsID(const char* fname, std::string ch, std::vector<int> noise_id = {}, int id_LT = 10000) {
	
	std::string cond;
	std::string data2plot;

	if (!(noise_id.empty())) {
		
		for (int i = 0; i < (noise_id.size()-1); i++) {
			cond.append("id != " + std::to_string(noise_id[i]) + "&& ");
		}
		cond.append("id != " + std::to_string(noise_id.back()) + "&& ");
	}

	cond.append("id < " + std::to_string(id_LT));
	data2plot.append("ch" + ch + ":id");

	TTree* v1751 = read_data(fname);
	v1751->Draw(data2plot.c_str(), cond.c_str(), "");
}

void plot_CHvsIT(const char* fname, std::string ch, std::vector<int> noise_id = {}, int id_LT = 10000) {
	
	std::string cond;
	std::string data2plot;

	if (!(noise_id.empty())) {
		
		for (int i = 0; i < (noise_id.size()-1); i++) {
			cond.append("id != " + std::to_string(noise_id[i]) + "&& ");
		}
		cond.append("id != " + std::to_string(noise_id.back()) + "&& ");
	}

	cond.append("id < " + std::to_string(id_LT));
	data2plot.append("ch" + ch + ":Iteration$");

	TTree* v1751 = read_data(fname);
	v1751->Draw(data2plot.c_str(), cond.c_str(), "LP");

}
