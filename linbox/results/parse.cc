#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>
using namespace std;

double get_sparsity (istream &ifs1, istream &ifs4, istream &ifs8){
	string line = "";
	while (line.find("sparsity M") == string::npos){
		getline(ifs1,line);
		getline(ifs4,line);
		getline(ifs8,line);
	}
	double sparsity = 1 ;
	string s;
	istringstream iss{line};
	iss >> s >> s;
	iss >> sparsity;
	return sparsity;
}

int get_D (istream &ifs1, istream &ifs4, istream &ifs8){
	string line;
	while(line.find("D:") == string::npos){
		getline(ifs1,line);
		getline(ifs4,line);
		getline(ifs8,line);
	}

	istringstream iss{line};
	iss >> line;
	int D;
	iss >> D;
	return D;
}

void advance_to_generic(istream &ifs1, istream &ifs4, istream &ifs8){
	string line = "";
	while (line.find("ENTER GENERIC LEX") == string::npos){
		getline(ifs1,line);
		getline(ifs4,line);
		getline(ifs8,line);
	}
}

int get_minpoly_deg (istream &ifs1, istream &ifs4, istream &ifs8){
	string line;
	while (line.find("min_poly deg") == string::npos){
		getline(ifs1,line);
		getline(ifs4,line);
		getline(ifs8,line);	
	}
	istringstream iss{line};
	iss >> line >> line >> line;
	int D;
	iss >> D;
	return D;
}

void get_real_time(double &t1, double &t4, double &t8, istream &ifs1, istream &ifs4, istream &ifs8){
	string line1, line4, line8;
	while (line1.find("Total real time") == string::npos){
		getline(ifs1,line1);
		getline(ifs4,line4);
		getline(ifs8,line8);
	}
	istringstream iss1{line1};
  istringstream iss4{line4};
	istringstream iss8{line8};
	iss1 >> line1 >> line1 >> line1 >> line1;
	iss4 >> line1 >> line1 >> line1 >> line1;
	iss8 >> line1 >> line1 >> line1 >> line1;
  iss1 >> t1;
	iss4 >> t4;
	iss8 >> t8;
}

void get_ls_time(double &t1, double &t4, double &t8, istream &ifs1, istream &ifs4, istream &ifs8){
	string line1, line4, line8;
	while (line1.find("left sequence (UT^i)") == string::npos){
		getline(ifs1,line1);
		getline(ifs4,line4);
		getline(ifs8,line8);
	}
	getline(ifs1,line1);
	getline(ifs4,line4);
	getline(ifs8,line8);


	istringstream iss1{line1};
  istringstream iss4{line4};
	istringstream iss8{line8};

	iss1 >> line1 >> line1 >> line1 >> line1;
	iss4 >> line1 >> line1 >> line1 >> line1;
	iss8 >> line1 >> line1 >> line1 >> line1;
  
	iss1 >> t1;
	iss4 >> t4;
	iss8 >> t8;
}

int main(int argc, char *argv[]){
	ifstream ifs1{string(argv[1]).append("1.out")};
	ifstream ifs4{string(argv[1]).append("2.out")};
	ifstream ifs8{string(argv[1]).append("4.out")};
	
	double sparsity = 0;
	double sparsity = get_sparsity(ifs1,ifs4,ifs8);
	int D = get_D(ifs1,ifs4,ifs8);

  double ngt1,ngt4,ngt8;
	get_real_time(ngt1,ngt4,ngt8,ifs1,ifs4,ifs8);

	advance_to_generic(ifs1,ifs4,ifs8);
	ofstream out{"compare-ng.tex", ios::app};
	double lt1,lt4,lt8;
	get_ls_time(lt1,lt4,lt8,ifs1,ifs4,ifs8);

	int min_poly_D = get_minpoly_deg(ifs1,ifs4,ifs8);

	double t1,t4,t8;
	get_real_time(t1,t4,t8,ifs1,ifs4,ifs8);

	out << argv[1] << "& &"<<D<<"&" << ngt1/t1 << "&" << ngt4/t4 << "&" << ngt8/t8 << min_poly_D << "/" << D  <<"\\\\\n" ;
	
	// handle output
cout << argv[1] << "& &" << D << "&" <<setprecision(3) << sparsity << "&" << (int)round(t1) << " (" << lt1/t1 << ")"
<< "&" << (int)round(t4) << " (" << lt4/t4 << ")" << "&" <<(int)round(t8) << " (" << lt8/t8 << ")"  << "&";
	if (D == min_poly_D) cout << "yes";
	else cout << "no";
	cout << "\\\\" << endl;

}
