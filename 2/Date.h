// Date.h
#ifndef DATE_H
#define DATE_H

#include <string>

using namespace std;

class Date {
public:
    Date(const string&);
private:
    unsigned int year;
    unsigned int month;
    unsigned int day;
};

Date::Date(const string& s) {
    string date = s;
    string::size_type i1 = 0, i2 = 0;
    if (s.find(',') != string::npos) {
		i1 = s.find(' ');
		i2 = s.find(',', i1+1);
		if(s.find("January") != string::npos) month = 1;
		if(s.find("February") != string::npos) month = 2;
		if(s.find("March") != string::npos) month = 3;
		if(s.find("April") != string::npos) month = 4;
		if(s.find("May") != string::npos) month = 5;
		if(s.find("June") != string::npos) month = 6;
		if(s.find("July") != string::npos) month = 7;
		if(s.find("August") != string::npos) month = 8;
		if(s.find("September") != string::npos) month = 9;
		if(s.find("October") != string::npos) month = 10;
		if(s.find("November") != string::npos) month = 11;
		if(s.find("December") != string::npos) month = 12;
		day = stoi(s.substr(i1+1, i2-i1-1));
		year = stoi(s.substr(i2+1, s.size()));
    }
    else if (s.find('/') != string::npos) {
		i1 = s.find('/');
		i2 = s.find('/', i1+1);
		month = stoi(s.substr(0, i1));
		day = stoi(s.substr(i1+1, i2-i1-1));
		year = stoi(s.substr(i2+1, s.size()));
    }
    else {
		i1 = s.find(' ');
		i2 = s.find(' ', i1+1);
		if(s.find("Jan") != string::npos)  month = 1;
		if(s.find("Feb") != string::npos)  month = 2;
		if(s.find("Mar") != string::npos)  month = 3;
		if(s.find("Apr") != string::npos)  month = 4;
		if(s.find("May") != string::npos)  month = 5;
		if(s.find("Jun") != string::npos)  month = 6;
		if(s.find("Jul") != string::npos)  month = 7;
		if(s.find("Aug") != string::npos)  month = 8;
		if(s.find("Sep") != string::npos)  month = 9;
		if(s.find("Oct") != string::npos)  month = 10;
		if(s.find("Nov") != string::npos)  month = 11;
		if(s.find("Dec") != string::npos)  month = 12;
		day = stoi(s.substr(i1+1, i2-i1-1));
		year = stoi(s.substr(i2+1, s.size()));
    }
}

#endif