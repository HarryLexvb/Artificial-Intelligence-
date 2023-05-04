//
// Created by HarryLex on 12/04/2023.
//

#include "pawn.h"

OwningPlayer otherPlayer(OwningPlayer current_player){
    if (current_player == HUMAN)
        return COMPUTER;
    return HUMAN;
}

Pawn::Pawn(int coord_x, int coord_y, float x_, float y_, OwningPlayer owner_): owner(owner_), x(x_), y(y_){
    coordinates.x = coord_x;
    coordinates.y = coord_y;
}

// void Pawn::lightUp(){

// }