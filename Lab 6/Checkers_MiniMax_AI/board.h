//
// Created by HarryLex on 12/04/2023.
//

#ifndef CHECKERS_MINIMAX_AI_BOARD_H
#define CHECKERS_MINIMAX_AI_BOARD_H

#include "pawn.h"

#include <SFML/Graphics.hpp>
#include <memory>
#include <vector>

const float board_size = 800; // size of the board in pixels
const float field_size = 77.5; // size of one field in pixels
const float border_size = 91; // size of the border in pixels

enum MoveType{
    INVALID, // invalid move
    NORMAL, // normal move
    BEAT, // beat move
    MULTI_BEAT, // multi beat move
};

struct Move{
    Move() = default; // default constructor
    Move(sf::Vector2i start_, sf::Vector2i finish_, MoveType type_): start(start_), finish(finish_), type(type_){}; // constructor
    sf::Vector2i start; // start coordinates
    sf::Vector2i finish; // finish coordinates
    MoveType type; // type of the move
};

struct Node{
    Node(Move move_, int value_): move(move_), value(value_){}; // constructor
    Move move; // move
    int value; // value of the move
};

class Board{
public:
    Board();
    ~Board();
    Board(const Board& copied);

    std::shared_ptr<Pawn> field[8][8]= {nullptr}; // 2D array of pointers to pawns
    std::vector<std::weak_ptr<Pawn>> pawn_vector; // vector of pointers to pawns
    std::vector<std::weak_ptr<Pawn>> player_pawns[2]; // vector of pointers to pawns for each player

    bool beat_possible [2] = {false}; // is beat possible for each player
    bool& getBeatPossible(OwningPlayer player); // get beat possible for player
    void resolveBeating(OwningPlayer player); // resolve beating for player

    std::vector<std::weak_ptr<Pawn>>& getVector (OwningPlayer player); // get vector of pawns for player
    std::shared_ptr<Pawn> getPawn(const sf::Vector2i& coords); // get pawn on given coordinates
    int setPawn(const sf::Vector2i& coords, const std::shared_ptr<Pawn>& new_ptr); // set pawn on given coordinates
    std::shared_ptr<Pawn> movePawn(sf::Vector2i start, sf::Vector2i finish, MoveType type); // move pawn from start to finish
    std::shared_ptr<Pawn> movePawn(const Move& move); // move pawn from start to finish
    MoveType checkMove(sf::Vector2i& start, sf::Vector2i& finish); // check if move is valid
    std::vector<Move>* getAvailibleMoves(OwningPlayer player); // get availible moves for player
    std::vector<Move>* getAvailibleMoves(OwningPlayer player, const std::shared_ptr<Pawn> pawn); // get availible moves for player
    int getScore (OwningPlayer player); // get score for player
    OwningPlayer checkWin(OwningPlayer player); // check if player won
    void print(); // print board
};


#endif //CHECKERS_MINIMAX_AI_BOARD_H
