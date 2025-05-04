#pragma once

#include <string>
#include <functional>
#include <iostream>

class MenuAction
{
public:

    MenuAction(
        std::wstring menu_name,
        wchar_t keyboard_option,
        int priority,
        std::function<bool()> action
    ):
        menu_name(menu_name),
        keyboard_option(keyboard_option),
        priority(priority),
        action(action)
    {};

    MenuAction() = delete;

    bool react()
    {
        return this->action();
    }

    wchar_t get_keyboard_option()
    {
        return this->keyboard_option;
    }

    int get_priority()
    {
        return this->priority;
    }

    std::wstring render()
    {
        std::wstring result = L"";

        result += L"[";
        result += this->keyboard_option;
        result += L"]";
        
        result += L" ";

        result += menu_name;
        
        return result;
    }

private:
    std::wstring menu_name;
    int priority;

    wchar_t keyboard_option;
    std::function<bool()> action;
};
