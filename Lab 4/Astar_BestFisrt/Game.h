//
// Created by HarryLex on 5/04/2023.
//

#ifndef ASTAR_BESTFISRT_GAME_H
#define ASTAR_BESTFISRT_GAME_H


#include <SFML\Graphics.hpp>
#include "Algorithm.h"
#include "Grid.h"
#include "Graph.h"
#include "AStar.h"
#include "BestFS.h"
#include "AlgorithmFactory.h"


class Game {			//game class. should have render, update/events, quit and initialize.

public:
    Game(int screenwidth, int screenheight, const std::string& title, int framerate);	//game constructor
    Game &operator=(Game &other) = delete;				//prevent assigning game object
    Game(const Game& other) = delete;					//no deep copying a game object
    Game(Game &&other) = delete;						//no moving game object
    ~Game();
    void update();					//update logic
    void render();					//rende r
    bool quit();					//quit the window

private:
    bool quitgame = false;			//quit game
    std::unique_ptr<Algorithm> algo;
    Grid grid;
    Graph graph;
    sf::Text text;
    sf::Font font;
    std::string graphtype = "Harold - Cledy - Massiel - Sergio";
    sf::Mouse mouse;
    sf::Event event;					//events class
    sf::RenderWindow createwindow;		//windows class
    bool solve = false;


};

#endif //ASTAR_BESTFISRT_GAME_H
