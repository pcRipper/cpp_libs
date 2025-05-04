#pragma once

#include <iostream>
#include <memory>

#include <conio.h>
#include <string>
#include <iostream>

#include <unordered_map>
#include <map>

#include <cli/menuAction.hpp>

class Menu
{
public:
    Menu() = default;
    Menu(Menu &menu) = delete;

    Menu(std::wstring title):
        title(title)
    {};

    Menu& add_option(std::shared_ptr<MenuAction> action)
    {
        menu[action->get_keyboard_option()] = action;
        sorted[action->get_priority()] = action->get_keyboard_option();

        return *this;
    }

    void run()
    {
        while(true)
        {
            system("cls");
            this->render();

            std::wcout << L"\nOption:";
            
            wchar_t input = _getch();

            system("cls");
            if(menu.count(input) == 0)
            {
                continue;
            }
            if(menu[input]->react())
            {
                break;
            }
        }
    }

    void render()
    {   
        if(0 < this->title.length())
        {
            std::wcout << L"   " << this->title << L"\n";
        }
        for(auto const [_, key] : this->sorted)
        {
            std::wcout << this->menu[key]->render() << L"\n";
        }
    }

private:
    std::unordered_map<wchar_t, std::shared_ptr<MenuAction>> menu;
    std::map<int, wchar_t> sorted;

    std::wstring title;
};
