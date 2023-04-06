//
// Created by HarryLex on 5/04/2023.
//

#ifndef ASTAR_BESTFISRT_BESTFS_H
#define ASTAR_BESTFISRT_BESTFS_H



#include "Algorithm.h"
#include "Graph.h"
#include "Grid.h"
#include <unordered_set>
#include <list>

class BestFS : public Algorithm {

public:
    BestFS(Graph &graph);
    virtual void SolveAlgorithm(const Location& srcpos, const Location& targetpos, const std::vector<Location>& obstacles, Grid &grid, sf::RenderWindow& createwindow) override;
    virtual void constructPath(Grid& grid) override;

private:
    Graph &graph;
    Location srcpos;
    Location targetpos;
    sf::Text text;
    sf::Font font;
    bool targetreached = false;
    std::list<node*> pq;
    std::unordered_set<node*> openSet;
    double nodedistance(node* a, node* b);
};


#endif //ASTAR_BESTFISRT_BESTFS_H
