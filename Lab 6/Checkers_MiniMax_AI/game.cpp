//
// Created by HarryLex on 12/04/2023.
//

#include "game.h"
#include <iostream>
#include <algorithm>

void delay(int miliseconds){
    sf::Clock clock;
    clock.restart();
    while(1){
        if(clock.getElapsedTime().asMilliseconds() > miliseconds)
            break;
    }
}

Game::Game(){
    //icon initialization
    icon.loadFromFile(R"(C:\Users\win 10\Documents\CLION\Checkers_MiniMax_AI\images\red.png)");
    //texture initialization
    textures[0].loadFromFile(R"(C:\Users\win 10\Documents\CLION\Checkers_MiniMax_AI\images\board.jpg)");
    textures[1].loadFromFile(R"(C:\Users\win 10\Documents\CLION\Checkers_MiniMax_AI\images\black.png)");
    textures[2].loadFromFile(R"(C:\Users\win 10\Documents\CLION\Checkers_MiniMax_AI\images\red.png)");
    textures[3].loadFromFile(R"(C:\Users\win 10\Documents\CLION\Checkers_MiniMax_AI\images\black_king.png)");
    textures[4].loadFromFile(R"(C:\Users\win 10\Documents\CLION\Checkers_MiniMax_AI\images\red_king.png)");
    //sprite initialization
    sprites[0].setTexture(textures[0]);
    for (int i = 1; i < 5; ++i){
        sprites[i].setTexture(textures[i]);
        sprites[i].setScale(0.6,0.6);
    }
}

void Game::start(){
    window.create(sf::VideoMode(board_size, board_size), "Miloszland-Checkers");
    window.setIcon(icon.getSize().x, icon.getSize().y, icon.getPixelsPtr());
    view();
}

int Game::manualMove(OwningPlayer player){
    sf::Vector2i mouse_position, start, finish;
    sf::Vector2i* updated_vector;
    std::shared_ptr<Pawn> active_pawn;
    bool mouse_pressed=false;
    while (window.isOpen()){
        mouse_pressed = pollEvents(mouse_position);
        if (mouse_pressed){
            if(mouse_position.x > border_size && mouse_position.x < board_size - border_size &&
               mouse_position.y > border_size && mouse_position.y < board_size - border_size){
                if (!active_pawn){
                    updated_vector = &start;
                }
                else{
                    updated_vector = &finish;
                }
                updated_vector->x = (mouse_position.x - border_size) / field_size;
                updated_vector->y = (mouse_position.y - border_size) / field_size;
                updated_vector->y = 7 - updated_vector->y;
                if (active_pawn){
                    //std::cerr << start.x << start.y << '-' << finish.x << finish.y << '\n';
                    if(active_pawn->owner == player){
                        MoveType result = game_board.checkMove(start, finish);
                        if (result != INVALID){
                            executeMove(start, finish, result);
                            return 0;
                        }
                    }
                    active_pawn = nullptr;
                }
                else {
                    active_pawn = game_board.getPawn(start);
                }
            }
        }
    }
    return 1;
}

void Game::play(){
    Move computer_move;
    OwningPlayer winner = NOBODY;
    while(winner == NOBODY){
        if(getMove(active_player))
            break;
        // std::cerr << alphabeta(game_board, computer_move, 2, COMPUTER, minus_infty, plus_infty);
        active_player = otherPlayer(active_player);
        winner = game_board.checkWin(active_player);
    }
    if (winner == HUMAN)
        std::cout << "Winn!!!\n";
    else if (winner == COMPUTER)
        std::cout << "LOOOSER :(\n";
}

int Game::computerMove(){
    Move computer_move;
    sf::Clock clock;
    clock.restart();
    alphabeta(game_board, computer_move, 6, COMPUTER, minus_infty, plus_infty);
    std::cerr << clock.getElapsedTime().asMilliseconds();
    executeMove(computer_move.start, computer_move.finish, computer_move.type);
    return 0;
}

int Game::getMove(OwningPlayer player){
    game_board.resolveBeating(player);
    if (player == COMPUTER)
        return computerMove();
    else
        return manualMove(HUMAN);
}

bool Game::pollEvents(sf::Vector2i& mouse_position){
    sf::Event event;
    while (window.pollEvent(event)){
        if (event.type == sf::Event::Closed){
            window.close();
            return false;
        }
        if (event.type == sf::Event::MouseButtonPressed){
            if (event.mouseButton.button == sf::Mouse::Left){
                mouse_position.x = event.mouseButton.x;
                mouse_position.y = event.mouseButton.y;
                return true;
            }
        }
    }
    return false;
}

void Game::view(){
    window.clear();
    //draw the board
    window.draw(sprites[0]);
    int sprite_number;
    //draw the pawns
    for(const auto pawn_ptr: game_board.pawn_vector){
        if (auto drawn_pawn = pawn_ptr.lock()){
            if (drawn_pawn->owner == HUMAN)
                sprite_number = 1;
            else
                sprite_number = 2;
            sprites[sprite_number].setPosition(drawn_pawn->x, drawn_pawn->y);
            window.draw(sprites[sprite_number]);
        }
    }
    window.display();
}

void Game::executeMove(sf::Vector2i& start, sf::Vector2i& finish, MoveType type){
    if(auto pawn = game_board.movePawn(start, finish, type)){
        float distance_x = ((finish.x - start.x) * field_size) / 10;
        float distance_y = ((finish-start).y * field_size) / 10;
        for (int i = 0; i < 10; ++i){
            pawn->x += distance_x;
            pawn->y -= distance_y;
            delay(20);
            view();
        }
        view();
    }

}

int Game::alphabeta(Board& current_board, Move& best_move, int depth, OwningPlayer player, int alpha, int beta){
    // std::cerr << "start poziom " << depth << '\n';
    int value; // value of the node
    // current_board.print();
    if (depth == 0){ //or node is a terminal node
        value = current_board.getScore(COMPUTER) - current_board.getScore(HUMAN); //heuristic function
        // std::cerr << " return " << value << '\n';
        return value; //return the heuristic value of the node
    }

    std::vector<Move>* possible_moves = current_board.getAvailibleMoves(player); //get all possible moves
    std::vector<Board>* possible_boards = new std::vector<Board>(possible_moves->size(), current_board); //create a vector of boards with the same state as current_board
    // std::cerr << possible_moves->size() << "dostępnych ruchów\n";
    for (unsigned int i = 0; i < possible_moves->size(); ++i){ //for each possible move
        possible_boards->at(i).movePawn(possible_moves->at(i)); //execute the move
    }
    if (player == COMPUTER){ //if it's computer's turn
        for (unsigned int i = 0; i < possible_boards->size(); ++i){ //for each possible board
            value = alphabeta(possible_boards->at(i), best_move, depth-1, HUMAN, alpha, beta); //call alphabeta on the board
            alpha = std::max(alpha, value); //update alpha
            if (alpha == value && depth == 6) //if the value is the best so far and it's the first level of recursion
                best_move = possible_moves->at(i); //save the move
            if (alpha >= beta){ //if alpha is greater or equal to beta
                // std::cerr << "alpha cut";
                break;
            }
        }
        return alpha; //return alpha
    }
    else{
        for (unsigned int i = 0; i < possible_boards->size(); ++i){ //for each possible board
            beta = std::min(beta, alphabeta(possible_boards->at(i), best_move, depth-1, COMPUTER, alpha, beta)); //call alphabeta on the board
            if (alpha >= beta){ //if alpha is greater or equal to beta
                // std::cerr << "beta cut";
                break; //beta cut
            }
        }
        return beta; //return beta
    }
    delete possible_moves;
    delete possible_boards;
    // std::cerr << "stop poziom " << depth << "- " << value << '\n';
}
