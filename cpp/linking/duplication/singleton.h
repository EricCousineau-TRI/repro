/*
 * singleton.h
 *
 *  Created on: 2011-12-24
 *      Author: bourneli
 */

#ifndef SINGLETON_H_
#define SINGLETON_H_

#include <iostream>

class singleton
{
public:
    singleton() {num = -1;}
    static singleton* pInstance;
public:
    static singleton& instance()
    {
        if (NULL == pInstance)
        {
            pInstance = new singleton();
        }
        static int count = 0;
        ++count;
        std::cout << "(count: " << count << ") ";
        return *pInstance;
    }
public:
    int num;
};

singleton* singleton::pInstance = NULL;

#endif /* SINGLETON_H_ */
