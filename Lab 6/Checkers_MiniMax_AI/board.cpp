//
// Created by HarryLex on 12/04/2023.
//

#include "board.h"
#include <iostream>

Board::Board(){	//pawn vector initialization
    pawn_vector.reserve(24); // 24 pawns
    float new_x, new_y; // coordinates of the pawn
    OwningPlayer new_player; // owner of the pawn
    std::shared_ptr<Pawn> new_ptr; // pointer to the pawn
    for (int i = 0; i < 8; ++i){ // 8 rows
        for (int j = 0; j < 8; ++j){ // 8 columns
            if (i%2 == j%2){ // if the field is black
                if (j < 3 || j > 4){ // if the field is not empty
                    new_x = border_size + i * field_size + 5; // coordinates of the pawn
                    new_y = border_size + (7-j) * field_size + 5; // coordinates of the pawn
                    if (j < 3) // owner of the pawn
                        new_player = HUMAN; // owner of the pawn
                    else if (j > 4) // owner of the pawn
                        new_player = COMPUTER; // owner of the pawn
                    new_ptr = std::make_shared<Pawn>(i, j, new_x, new_y, new_player); // pointer to the pawn
                    field[i][j] = new_ptr; // pointer to the pawn
                    pawn_vector.push_back(std::weak_ptr<Pawn>(new_ptr));    // pointer to the pawn
                    getVector(new_player).push_back(std::weak_ptr<Pawn>(new_ptr)); // pointer to the pawn
                }
            }
        }
    }
}

Board::~Board(){}

Board::Board(const Board& copied){ // copy constructor
    for (int i = 0; i < 8; ++i){  // 8 rows
        for (int j = 0; j < 8; ++j){ // 8 columns
            if (copied.field[i][j]){ // if the field is not empty
                std::shared_ptr<Pawn> new_ptr = std::shared_ptr<Pawn>(new Pawn(*copied.field[i][j])); // pointer to the pawn
                field[i][j] = new_ptr; // pointer to the pawn
                pawn_vector.push_back(std::weak_ptr<Pawn>(new_ptr));    // pointer to the pawn
                getVector(new_ptr->owner).push_back(std::weak_ptr<Pawn>(new_ptr)); // pointer to the pawn

            }
        }
    }
}

std::vector<std::weak_ptr<Pawn>>& Board::getVector (OwningPlayer player){ // returns the vector of pawns of the player
    if (player == HUMAN) // returns the vector of pawns of the player
        return player_pawns[0]; // returns the vector of pawns of the player
    else
        return player_pawns[1]; // returns the vector of pawns of the player
}

std::shared_ptr<Pawn> Board::getPawn(const sf::Vector2i& coords){ // returns the pointer to the pawn on the field
    if (field[coords.x][coords.y] != nullptr) // returns the pointer to the pawn on the field
        return field[coords.x][coords.y]; // returns the pointer to the pawn on the field
    else
        return nullptr; // returns the pointer to the pawn on the field
}

int Board::setPawn(const sf::Vector2i& coords, const std::shared_ptr<Pawn>& new_ptr){ // sets the pointer to the pawn on the field
    field[coords.x][coords.y] = new_ptr; // sets the pointer to the pawn on the field
    return 0;
}

void Board::print(){ // prints the board
    for (int y = 7; y > -1 ; --y){ // 8 rows
        for (int x = 0; x < 8; ++x){ // 8 columns
            auto printed_pawn = getPawn(sf::Vector2i(x, y)); // pointer to the pawn
            if (printed_pawn){ // if the field is not empty
                if (printed_pawn->owner == HUMAN) // if the pawn is human
                    std::cerr << 'O'; // if the pawn is human
                else
                    std::cerr << 'X'; // if the pawn is computer
            }
            else
                std::cerr << ' '; // if the field is empty
        }
        std::cerr << '\n';
    }
    std::cerr << '\n';
}

std::shared_ptr<Pawn> Board::movePawn(sf::Vector2i start, sf::Vector2i finish, MoveType type){ // moves the pawn
    if (auto pawn = getPawn(start)){ // moves the pawn
        int direction = 1; // moves the pawn
        if (pawn->owner == COMPUTER) // moves the pawn
            direction = -1; // moves the pawn
        if(type == BEAT){  // moves the pawn
            sf::Vector2i beaten_pawn(start.x + (finish.x - start.x)/2, start.y + direction); // moves the pawn
            getPawn(beaten_pawn).reset(); // moves the pawn
            setPawn(beaten_pawn, nullptr); // moves the pawn
        }
        setPawn(start, nullptr);    // moves the pawn
        setPawn(finish, pawn);  // moves the pawn
        pawn->coordinates = finish; // moves the pawn
        resolveBeating(pawn->owner);    // moves the pawn

        return pawn;   // moves the pawn
    }
    return nullptr;
}

std::shared_ptr<Pawn> Board::movePawn(const Move& move){ // moves the pawn
    return movePawn(move.start, move.finish, move.type); // moves the pawn
}

MoveType Board::checkMove(sf::Vector2i& start, sf::Vector2i& finish){ // checks the move
    MoveType result = INVALID; // checks the move
    if(finish.x >= 0 && finish.x <= 7 && finish.y >= 0 && finish.y <=7){ // checks the move
        if(std::shared_ptr<Pawn> pawn = getPawn(start)){ // checks the move
            int direction = 1; // checks the move
            if (pawn->owner == COMPUTER) // checks the move
                direction = -1; // checks the move
            if (finish.y == start.y + direction){
                if (finish.x == start.x + 1 || finish.x == start.x - 1){ // checks the move
                    if (!getPawn(finish)){ // checks the move
                        if(!getBeatPossible(pawn->owner)){ // checks the move
                            result = NORMAL; // checks the move
                        }
                    }
                }
            }
            else if (finish.y == start.y + 2*direction){ // checks the move
                if (finish.x == start.x + 2 || finish.x == start.x - 2){ // checks the move
                    if (!getPawn(finish)){ // checks the move
                        sf::Vector2i beaten_pawn(start.x + (finish.x - start.x)/2, start.y + direction); // checks the move
                        if (getPawn(beaten_pawn)){ // checks the move
                            if (getPawn(beaten_pawn)->owner == otherPlayer(pawn->owner)){ // checks the move
                                result = BEAT; // checks the move
                            }
                        }
                    }
                }
            }
        }
    }
    return result; // checks the move
}

bool& Board::getBeatPossible(OwningPlayer player){ // returns the beat_possible variable
    if (player == COMPUTER) // returns the beat_possible variable
        return beat_possible[1]; // returns the beat_possible variable
    return beat_possible[0]; // returns the beat_possible variable
}

void Board::resolveBeating(OwningPlayer player){
    getBeatPossible(player) = false;
    std::vector<Move>* move_vector = getAvailibleMoves(player);
    for (auto tested_move: *move_vector){
        if (tested_move.type == BEAT)
            getBeatPossible(player) = true;
    }
}

std::vector<Move>* Board::getAvailibleMoves(OwningPlayer player){
    std::vector<Move>* move_vector = new std::vector<Move>;
    for (auto pawn_ptr: getVector(player)){
        if (auto pawn = pawn_ptr.lock()){
            auto new_moves = getAvailibleMoves(player, pawn);
            if(!new_moves->empty())
                move_vector->insert(move_vector->end(), new_moves->begin(), new_moves->end());
            delete new_moves;
        }
    }
    return move_vector;
}

std::vector<Move>* Board::getAvailibleMoves(OwningPlayer player, const std::shared_ptr<Pawn> pawn){
    std::vector<Move>* move_vector = new std::vector<Move>;
    sf::Vector2i start, finish;
    int direction = 1;
    if (pawn){
        if (player == COMPUTER)
            direction = -1;
        start = pawn->coordinates;
        for (int k: {1,2}){
            for (int l: {-1,1}){
                finish = start + sf::Vector2i(l*k, k*direction);
                // std::cerr << start.x << ' ' << start.y << ' ' << finish.x <<  ' ' << finish.y;
                MoveType result = checkMove(start, finish);
                if (result != INVALID){
                    Move new_move = Move(start, finish, result);
                    // std::cerr << " valid move";
                    move_vector->push_back(new_move);
                }
                // std::cerr << '\n';
            }
        }
    }
    return move_vector;
}

int Board::getScore(OwningPlayer player){
    int score = 0;
    for (auto pawn_weak: getVector(player)){
        if (auto pawn = pawn_weak.lock()){
            std::vector<Move>* move_vector = new std::vector<Move>;
            int x = pawn->coordinates.x;
            int y = pawn->coordinates.y;
            score += 10;
            if (player == HUMAN){
                if (y == 2 || y == 3)
                    score += 1;
                else if (y == 4 || y == 5)
                    score += 3;
                else if (y == 6 || y == 7)
                    score += 5;
            }
            else{
                if (y == 5 || y == 4)
                    score += 1;
                else if (y == 3 || y == 2)
                    score += 3;
                else if (y == 1 || y == 0)
                    score += 5;
            }
            if ((x == 0 || x == 7) && (y == 0 || y == 7))
                score += 2;
            else if ((x == 1 || x == 6) && (y == 1 || y == 6))
                score += 1;
            move_vector = getAvailibleMoves(player, pawn);
            if (!move_vector->empty()){
                for (auto tested_move: *move_vector){
                    if (tested_move.type == BEAT)
                        score += 30;
                }
            }
            delete move_vector;
        }
    }
    return score;
}

OwningPlayer Board::checkWin(OwningPlayer player){
    resolveBeating(player);
    OwningPlayer winner = NOBODY;
    std::vector<Move>* availible_moves;

    availible_moves = getAvailibleMoves(player);
    if (availible_moves->empty()){
        winner = otherPlayer(player);
    }
    else {
        int pawn_count = 0;
        for (auto checked_pawn: getVector(player)){
            if (!checked_pawn.expired())
                ++pawn_count;
        }
        if(!pawn_count){
            winner = otherPlayer(player);
        }
    }
    delete availible_moves;
    return winner;
}
