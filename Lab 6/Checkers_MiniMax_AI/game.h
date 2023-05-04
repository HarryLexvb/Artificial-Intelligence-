//
// Created by HarryLex on 12/04/2023.
//

#ifndef CHECKERS_MINIMAX_AI_GAME_H
#define CHECKERS_MINIMAX_AI_GAME_H
#include <vector>
#include <memory>
#include <SFML/Graphics.hpp>

#include "pawn.h"
#include "board.h"

const int plus_infty = 10000;
const int minus_infty = -10000;

class Game{
public:
    Game(); // constructor
    void start(); // start game
    void play(); // play game
    void end(); // end game
    void view(); // view the board
    int getMove(OwningPlayer player); // get move
    int manualMove(OwningPlayer player); // manual move
    int computerMove(); // computer move
    void executeMove(sf::Vector2i& start, sf::Vector2i& finish, MoveType type); // execute move
    bool pollEvents(sf::Vector2i& mouse_position); // poll events
    int alphabeta(Board&, Move&, int depth, OwningPlayer, int alpha, int beta); // alpha-beta pruning algorithm

    sf::RenderWindow window; // window
    sf::Texture textures[5]; // textures
    sf::Sprite sprites[5]; // sprites
    sf::Image icon; // icon

    OwningPlayer players[2] = {HUMAN, COMPUTER}; // players
    Board game_board; // game board
    OwningPlayer active_player = HUMAN; // active player
};

#endif //CHECKERS_MINIMAX_AI_GAME_H
