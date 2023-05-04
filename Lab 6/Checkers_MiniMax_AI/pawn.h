//
// Created by HarryLex on 12/04/2023.
//

#ifndef CHECKERS_MINIMAX_AI_PAWN_H
#define CHECKERS_MINIMAX_AI_PAWN_H

#include <SFML/Graphics.hpp>

enum OwningPlayer{ // enum for player
    NOBODY, // no player
    HUMAN, // human player
    COMPUTER, // computer player
};

OwningPlayer otherPlayer(OwningPlayer current_player); // function for getting other player

enum PawnLevel{ // enum for pawn level
    normal, // normal pawn
    king, // king pawn
};

class Pawn{ // class for pawn
public:
    Pawn(int, int, float, float, OwningPlayer); // constructor
    void lightUp(); // light up pawn
    OwningPlayer owner; // owner of pawn
    PawnLevel level = normal; // level of pawn
    float x; // x coordinate
    float y; // y coordinate
    sf::Vector2i coordinates; // coordinates
};

#endif //CHECKERS_MINIMAX_AI_PAWN_H
