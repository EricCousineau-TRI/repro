/*
 * singleton.h
 *
 *  Created on: 2011-12-24
 *      Author: bourneli
 */

#ifndef SINGLETON_H_
#define SINGLETON_H_

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
        return *pInstance;
    }
public:
    int num;
};

singleton* singleton::pInstance = NULL;

#endif /* SINGLETON_H_ */
