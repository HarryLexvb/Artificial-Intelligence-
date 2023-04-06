#include <iostream>
#include "Game.h"

int main()
{

    Game creategame(700, 700, "graph visualizer", 120);

    while (!creategame.quit()) {
        creategame.update();
        creategame.render();
    }
}