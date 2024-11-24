
#include "mapgen_generate.h"

#include <cmath>
#include <queue>
#include <map>
#include <algorithm>
// #include <iostream>


namespace map_gen
{

// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());

bool Leaf::split()
{
    if (child1 || child2)
        return false;

    bool splitH = std::uniform_real_distribution<>(0, 1)(gen) > 0.5;
    if (w > h && w / h >= 1.25)
        splitH = false;
    else if (h > w && h / w >= 1.25)
        splitH = true;

    int max = (splitH ? h : w) - MIN_LEAF_SIZE;
    if (max <= MIN_LEAF_SIZE)
        return false;

    int split = std::uniform_int_distribution<>(MIN_LEAF_SIZE, max)(gen);

    if (splitH)
    {
        child1 = std::make_unique<Leaf>(x, y, w, split);
        child2 = std::make_unique<Leaf>(x, y + split, w, h - split);
    }
    else
    {
        child1 = std::make_unique<Leaf>(x, y, split, h);
        child2 = std::make_unique<Leaf>(x + split, y, w - split, h);
    }

    return true;
}

void TunnelingGenerator::generate()
{
    clearMap();
    rooms_.clear();

    for (int i = 0; i < MAX_ROOMS; i++)
    {
        int w = std::uniform_int_distribution<>(ROOM_MIN_SIZE, ROOM_MAX_SIZE)(gen);
        int h = std::uniform_int_distribution<>(ROOM_MIN_SIZE, ROOM_MAX_SIZE)(gen);
        int x = std::uniform_int_distribution<>(1, width_ - w - 1)(gen);
        int y = std::uniform_int_distribution<>(1, height_ - h - 1)(gen);

        Room newRoom(x, y, w, h);

        bool failed = false;
        for (const auto &otherRoom : rooms_)
        {
            if (newRoom.intersect(otherRoom))
            {
                failed = true;
                break;
            }
        }

        if (!failed)
        {
            createRoom(newRoom);
            if (!rooms_.empty())
            {
                auto [prevCenterX, prevCenterY] = rooms_.back().center();
                auto [newCenterX, newCenterY] = newRoom.center();

                if (std::uniform_real_distribution<>(0, 1)(gen) > 0.5)
                {
                    createHorizontalTunnel(prevCenterX, newCenterX, prevCenterY);
                    createVerticalTunnel(prevCenterY, newCenterY, newCenterX);
                }
                else
                {
                    createVerticalTunnel(prevCenterY, newCenterY, prevCenterX);
                    createHorizontalTunnel(prevCenterX, newCenterX, newCenterY);
                }
            }
            rooms_.push_back(newRoom);
        }
    }
}

void TunnelingGenerator::createRoom(const Room &room)
{
    for (int x = room.x1 + 1; x < room.x2; x++)
    {
        for (int y = room.y1 + 1; y < room.y2; y++)
        {
            if (x > 0 && x < width_ && y > 0 && y < height_)
            {
                map_[x][y] = 0;
            }
        }
    }
}

void TunnelingGenerator::createHorizontalTunnel(int x1, int x2, int y)
{
    for (int x = std::min(x1, x2); x <= std::max(x1, x2); x++)
    {
        if (x > 0 && x < width_ && y > 0 && y < height_)
        {
            map_[x][y] = 0;
        }
    }
}

void TunnelingGenerator::createVerticalTunnel(int y1, int y2, int x)
{
    for (int y = std::min(y1, y2); y <= std::max(y1, y2); y++)
    {
        if (x > 0 && x < width_ && y > 0 && y < height_)
        {
            map_[x][y] = 0;
        }
    }
}

void BSPGenerator::generate()
{
    leafs_.reserve(512);
    clearMap();
    leafs_.clear();

    auto rootLeaf = std::make_unique<Leaf>(0, 0, width_, height_);
    leafs_.push_back(rootLeaf.get());

    bool splitSuccessfully = true;
    while (splitSuccessfully)
    {
        splitSuccessfully = false;
        for (auto leaf : leafs_)
        {
            if (!leaf->child1 && !leaf->child2)
            {
                if (leaf->w > MAX_LEAF_SIZE ||
                    leaf->h > MAX_LEAF_SIZE ||
                    std::uniform_real_distribution<>(0, 1)(gen) > 0.8)
                {
                    if (leaf->split())
                    {
                        leafs_.push_back(leaf->child1.get());
                        leafs_.push_back(leaf->child2.get());
                        splitSuccessfully = true;
                    }
                }
            }
        }
    }

    createRooms();
}

void BSPGenerator::createRooms()
{
    for (auto &leaf : leafs_)
    {
        if (!leaf->child1 && !leaf->child2)
        {
            int w = std::uniform_int_distribution<>(ROOM_MIN_SIZE,
                                                    std::min(ROOM_MAX_SIZE, leaf->w - 1))(gen);
            int h = std::uniform_int_distribution<>(ROOM_MIN_SIZE,
                                                    std::min(ROOM_MAX_SIZE, leaf->h - 1))(gen);
            int x = std::uniform_int_distribution<>(
                leaf->x, leaf->x + (leaf->w - 1) - w)(gen);
            int y = std::uniform_int_distribution<>(
                leaf->y, leaf->y + (leaf->h - 1) - h)(gen);

            leaf->room = std::make_unique<Room>(x, y, w, h);
            createRoom(*leaf->room);
        }
    }

    // Connect rooms
    for (size_t i = 0; i < leafs_.size() - 1; i++)
    {
        if (leafs_[i]->room && leafs_[i + 1]->room)
        {
            connectRooms(*leafs_[i]->room, *leafs_[i + 1]->room);
        }
    }
}

void BSPGenerator::createRoom(const Room &room)
{
    for (int x = room.x1 + 1; x < room.x2; x++)
    {
        for (int y = room.y1 + 1; y < room.y2; y++)
        {
            if (x > 0 && x < width_ && y > 0 && y < height_)
            {
                map_[x][y] = 0;
            }
        }
    }
}

void BSPGenerator::connectRooms(const Room &room1, const Room &room2)
{
    auto [x1, y1] = room1.center();
    auto [x2, y2] = room2.center();

    if (std::uniform_real_distribution<>(0, 1)(gen) > 0.5)
    {
        createHorizontalTunnel(x1, x2, y1);
        createVerticalTunnel(y1, y2, x2);
    }
    else
    {
        createVerticalTunnel(y1, y2, x1);
        createHorizontalTunnel(x1, x2, y2);
    }
}

void BSPGenerator::createHorizontalTunnel(int x1, int x2, int y)
{
    for (int x = std::min(x1, x2); x <= std::max(x1, x2); x++)
    {
        if (x > 0 && x < width_ && y > 0 && y < height_)
        {
            map_[x][y] = 0;
        }
    }
}

void BSPGenerator::createVerticalTunnel(int y1, int y2, int x)
{
    for (int y = std::min(y1, y2); y <= std::max(y1, y2); y++)
    {
        if (x > 0 && x < width_ && y > 0 && y < height_)
        {
            map_[x][y] = 0;
        }
    }
}

void DrunkardWalkGenerator::generate()
{
    clearMap();
    filled_ = 0;
    previousDirection_ = "";

    drunkardX_ = std::uniform_int_distribution<>(2, width_ - 3)(gen);
    drunkardY_ = std::uniform_int_distribution<>(2, height_ - 3)(gen);

    int filledGoal = width_ * height_ * PERCENT_GOAL;

    for (int i = 0; i < WALK_ITERATIONS && filled_ < filledGoal; i++)
    {
        walk();
    }
}

void DrunkardWalkGenerator::walk()
{
    float north = 1.0f;
    float south = 1.0f;
    float east = 1.0f;
    float west = 1.0f;

    // Weight against edges
    if (drunkardX_ < width_ * 0.25)
    {
        east += WEIGHTED_TOWARD_CENTER;
    }
    else if (drunkardX_ > width_ * 0.75)
    {
        west += WEIGHTED_TOWARD_CENTER;
    }
    if (drunkardY_ < height_ * 0.25)
    {
        south += WEIGHTED_TOWARD_CENTER;
    }
    else if (drunkardY_ > height_ * 0.75)
    {
        north += WEIGHTED_TOWARD_CENTER;
    }

    // Weight toward previous direction
    if (previousDirection_ == "north")
        north += WEIGHTED_TOWARD_PREVIOUS_DIRECTION;
    if (previousDirection_ == "south")
        south += WEIGHTED_TOWARD_PREVIOUS_DIRECTION;
    if (previousDirection_ == "east")
        east += WEIGHTED_TOWARD_PREVIOUS_DIRECTION;
    if (previousDirection_ == "west")
        west += WEIGHTED_TOWARD_PREVIOUS_DIRECTION;

    // Normalize probabilities
    float total = north + south + east + west;
    north /= total;
    south /= total;
    east /= total;
    west /= total;

    // Choose direction
    float choice = std::uniform_real_distribution<>(0, 1)(gen);
    int dx = 0, dy = 0;

    if (choice < north)
    {
        dy = -1;
        previousDirection_ = "north";
    }
    else if (choice < north + south)
    {
        dy = 1;
        previousDirection_ = "south";
    }
    else if (choice < north + south + east)
    {
        dx = 1;
        previousDirection_ = "east";
    }
    else
    {
        dx = -1;
        previousDirection_ = "west";
    }

    // Move drunkard
    if (drunkardX_ + dx > 0 && drunkardX_ + dx < width_ - 1 &&
        drunkardY_ + dy > 0 && drunkardY_ + dy < height_ - 1)
    {
        drunkardX_ += dx;
        drunkardY_ += dy;
        if (map_[drunkardX_][drunkardY_] == 1)
        {
            map_[drunkardX_][drunkardY_] = 0;
            filled_++;
        }
    }
}

void CellularAutomataGenerator::generate()
{
    clearMap();
    caves_.clear();

    // Random fill
    randomFillMap();
    // Create caves
    createCaves();
    // Get and connect caves
    getCaves();
    connectCaves();

    // Clean up
    cleanUpMap();
}

void CellularAutomataGenerator::randomFillMap()
{
    for (int y = 1; y < height_ - 1; y++)
    {
        for (int x = 1; x < width_ - 1; x++)
        {
            if (std::uniform_real_distribution<>(0, 1)(gen) >= WALL_PROBABILITY)
            {
                map_[x][y] = 0;
            }
        }
    }
}

void CellularAutomataGenerator::createCaves()
{
    for (int i = 0; i < ITERATIONS; i++)
    {
        int tileX = std::uniform_int_distribution<>(1, width_ - 2)(gen);
        int tileY = std::uniform_int_distribution<>(1, height_ - 2)(gen);

        if (getAdjacentWalls(tileX, tileY) > NEIGHBORS)
        {
            map_[tileX][tileY] = 1;
        }
        else if (getAdjacentWalls(tileX, tileY) < NEIGHBORS)
        {
            map_[tileX][tileY] = 0;
        }
    }
    cleanUpMap();
}

void CellularAutomataGenerator::cleanUpMap()
{
    if (SMOOTH_EDGES)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int x = 1; x < width_ - 1; x++)
            {
                for (int y = 1; y < height_ - 1; y++)
                {
                    if (map_[x][y] == 1 && getAdjacentWallsSimple(x, y) <= SMOOTHING)
                    {
                        map_[x][y] = 0;
                    }
                }
            }
        }
    }
}

void CellularAutomataGenerator::createTunnel(
    const std::set<std::pair<int, int>> &cave1,
    const std::pair<int, int> &point1,
    const std::pair<int, int> &point2)
{
    int x = point2.first;
    int y = point2.second;

    while (true)
    {
        if (cave1.find(std::make_pair(x, y)) != cave1.end())
        {
            break;
        }

        // Calculate weights
        double north = 1.0, south = 1.0, east = 1.0, west = 1.0;
        const double weight = 1.0;

        if (x < point1.first)
            east += weight;
        else if (x > point1.first)
            west += weight;
        if (y < point1.second)
            south += weight;
        else if (y > point1.second)
            north += weight;

        // Normalize weights
        double total = north + south + east + west;
        north /= total;
        south /= total;
        east /= total;
        west /= total;

        // Choose direction
        double choice = std::uniform_real_distribution<>(0, 1)(gen);
        int dx = 0, dy = 0;

        if (choice < north)
        {
            dy = -1;
        }
        else if (choice < north + south)
        {
            dy = 1;
        }
        else if (choice < north + south + east)
        {
            dx = 1;
        }
        else
        {
            dx = -1;
        }

        // Check bounds and move
        if (x + dx > 0 && x + dx < width_ - 1 && y + dy > 0 && y + dy < height_ - 1)
        {
            x += dx;
            y += dy;
            map_[x][y] = 0;
        }
    }
}

int CellularAutomataGenerator::getAdjacentWallsSimple(int x, int y)
{
    int wallCount = 0;
    if (map_[x][y - 1] == 1)
        wallCount++;
    if (map_[x][y + 1] == 1)
        wallCount++;
    if (map_[x - 1][y] == 1)
        wallCount++;
    if (map_[x + 1][y] == 1)
        wallCount++;
    return wallCount;
}

int CellularAutomataGenerator::getAdjacentWalls(int x, int y)
{
    int wallCount = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue;
            int checkX = x + i;
            int checkY = y + j;
            if (checkX > 0 && checkX < width_ && checkY > 0 && checkY < height_)
            {
                wallCount += map_[checkX][checkY];
            }
        }
    }
    return wallCount;
}

void CellularAutomataGenerator::getCaves()
{
    for (int x = 0; x < width_; x++)
    {
        for (int y = 0; y < height_; y++)
        {
            if (map_[x][y] == 0)
            {
                floodFill(x, y);
            }
        }
    }

    for (const auto &cave : caves_)
    {
        for (const auto &tile : cave)
        {
            map_[tile.first][tile.second] = 0;
        }
    }
}

void CellularAutomataGenerator::floodFill(int startX, int startY)
{
    std::set<std::pair<int, int>> cave;
    std::set<std::pair<int, int>> toBeFilled;
    toBeFilled.insert({startX, startY});

    while (!toBeFilled.empty())
    {
        auto tile = *toBeFilled.begin();
        toBeFilled.erase(toBeFilled.begin());

        if (cave.find(tile) == cave.end())
        {
            cave.insert(tile);
            map_[tile.first][tile.second] = 1;

            // Check adjacent cells
            const std::vector<std::pair<int, int>> directions = {
                {0, -1}, {0, 1}, {1, 0}, {-1, 0}};

            for (const auto &dir : directions)
            {
                int newX = tile.first + dir.first;
                int newY = tile.second + dir.second;
                if (newX > 0 && newX < width_ && newY > 0 && newY < height_)
                {
                    if (map_[newX][newY] == 0)
                    {
                        std::pair<int, int> newTile = {newX, newY};
                        if (toBeFilled.find(newTile) == toBeFilled.end() &&
                            cave.find(newTile) == cave.end())
                        {
                            toBeFilled.insert(newTile);
                        }
                    }
                }
            }
        }
    }

    if (cave.size() >= ROOM_MIN_SIZE)
    {
        caves_.push_back(cave);
    }
}

void CellularAutomataGenerator::connectCaves()
{
    for (auto &currentCave : caves_)
    {
        std::pair<int, int> point1 = *currentCave.begin();
        std::pair<int, int> point2;
        const std::set<std::pair<int, int>> *targetCave = nullptr;
        double minDistance = std::numeric_limits<double>::max();

        for (auto &nextCave : caves_)
        {
            if (&nextCave != &currentCave && !checkConnectivity(currentCave, nextCave))
            {
                std::pair<int, int> nextPoint = *nextCave.begin();
                double dist = distanceFormula(point1, nextPoint);
                if (dist < minDistance)
                {
                    point2 = nextPoint;
                    minDistance = dist;
                    targetCave = &nextCave;
                }
            }
        }

        if (targetCave)
        {
            createTunnel(currentCave, point1, point2);
        }
    }
}

bool CellularAutomataGenerator::checkConnectivity(
    const std::set<std::pair<int, int>> &cave1,
    const std::set<std::pair<int, int>> &cave2)
{
    std::set<std::pair<int, int>> connected;
    std::set<std::pair<int, int>> toCheck;
    toCheck.insert(*cave1.begin());

    while (!toCheck.empty())
    {
        auto current = *toCheck.begin();
        toCheck.erase(toCheck.begin());

        if (connected.find(current) == connected.end())
        {
            connected.insert(current);

            // Check adjacent cells
            const std::vector<std::pair<int, int>> dirs = {
                {0, -1}, {0, 1}, {1, 0}, {-1, 0}};

            for (const auto &dir : dirs)
            {
                int newX = current.first + dir.first;
                int newY = current.second + dir.second;
                std::pair<int, int> next{newX, newY};

                if (map_[newX][newY] == 0 &&
                    toCheck.find(next) == toCheck.end() &&
                    connected.find(next) == connected.end())
                {
                    toCheck.insert(next);
                }
            }
        }
    }

    return connected.find(*cave2.begin()) != connected.end();
}

double CellularAutomataGenerator::distanceFormula(
    const std::pair<int, int> &p1,
    const std::pair<int, int> &p2)
{
    return std::sqrt(std::pow(p2.first - p1.first, 2) +
                     std::pow(p2.second - p1.second, 2));
}

void RoomAdditionGenerator::generate()
{
    clearMap();
    rooms_.clear();

    // Generate first room
    auto firstRoom = generateRoom();
    auto [roomWidth, roomHeight] = getRoomDimensions(firstRoom);
    int roomX = (width_ / 2 - roomWidth / 2) - 1;
    int roomY = (height_ / 2 - roomHeight / 2) - 1;
    addRoom(roomX, roomY, firstRoom);

    // Generate other rooms
    for (int i = 0; i < BUILD_ROOM_ATTEMPTS; i++)
    {
        auto room = generateRoom();
        auto [x, y, wallTile, direction, length] = placeRoom(room);

        if (x != -1 && y != -1)
        {
            addRoom(x, y, room);
            addTunnel(wallTile, direction, length);

            if (rooms_.size() >= MAX_NUM_ROOMS)
            {
                break;
            }
        }
    }
}

std::vector<std::vector<int>> RoomAdditionGenerator::generateRoom()
{
    if (!rooms_.empty())
    {
        float choice = std::uniform_real_distribution<>(0, 1)(gen);

        if (choice < SQUARE_ROOM_CHANCE)
        {
            return generateSquareRoom();
        }
        else if (choice < SQUARE_ROOM_CHANCE + CROSS_ROOM_CHANCE)
        {
            return generateCrossRoom();
        }
        else
        {
            return generateCellularRoom();
        }
    }
    else
    {
        if (std::uniform_real_distribution<>(0, 1)(gen) < CAVERN_CHANCE)
        {
            return generateCavernRoom();
        }
        else
        {
            return generateSquareRoom();
        }
    }
}

std::vector<std::vector<int>> RoomAdditionGenerator::generateSquareRoom()
{
    int width = std::uniform_int_distribution<>(
        SQUARE_ROOM_MIN_SIZE, SQUARE_ROOM_MAX_SIZE)(gen);
    int height = std::uniform_int_distribution<>(
        std::max(int(width * 0.5), SQUARE_ROOM_MIN_SIZE),
        std::min(int(width * 1.5), SQUARE_ROOM_MAX_SIZE))(gen);

    std::vector<std::vector<int>> room(width, std::vector<int>(height, 1));

    // Create interior
    for (int x = 1; x < width - 1; x++)
    {
        for (int y = 1; y < height - 1; y++)
        {
            room[x][y] = 0;
        }
    }

    return room;
}

std::vector<std::vector<int>> RoomAdditionGenerator::generateCrossRoom()
{
    int horWidth = ((std::uniform_int_distribution<>(
                        CROSS_ROOM_MIN_SIZE + 2, CROSS_ROOM_MAX_SIZE)(gen)) /
                    2) *
                   2;
    int virHeight = ((std::uniform_int_distribution<>(
                         CROSS_ROOM_MIN_SIZE + 2, CROSS_ROOM_MAX_SIZE)(gen)) /
                     2) *
                    2;
    int horHeight = ((std::uniform_int_distribution<>(
                         CROSS_ROOM_MIN_SIZE, virHeight - 2)(gen)) /
                     2) *
                    2;
    int virWidth = ((std::uniform_int_distribution<>(
                        CROSS_ROOM_MIN_SIZE, horWidth - 2)(gen)) /
                    2) *
                   2;

    std::vector<std::vector<int>> room(horWidth, std::vector<int>(virHeight, 1));

    // Fill horizontal space
    int virOffset = (virHeight / 2 - horHeight / 2);
    for (int y = virOffset; y < horHeight + virOffset; y++)
    {
        for (int x = 0; x < horWidth; x++)
        {
            room[x][y] = 0;
        }
    }

    // Fill vertical space
    int horOffset = (horWidth / 2 - virWidth / 2);
    for (int y = 0; y < virHeight; y++)
    {
        for (int x = horOffset; x < virWidth + horOffset; x++)
        {
            room[x][y] = 0;
        }
    }

    return room;
}

std::vector<std::vector<int>> RoomAdditionGenerator::generateCellularRoom()
{
    while (true)
    {
        std::vector<std::vector<int>> room(
            ROOM_MAX_SIZE, std::vector<int>(ROOM_MAX_SIZE, 1));

        // Random fill
        for (int y = 2; y < ROOM_MAX_SIZE - 2; y++)
        {
            for (int x = 2; x < ROOM_MAX_SIZE - 2; x++)
            {
                if (std::uniform_real_distribution<>(0, 1)(gen) >= WALL_PROBABILITY)
                {
                    room[x][y] = 0;
                }
            }
        }

        // Cellular automata iterations
        for (int i = 0; i < 4; i++)
        {
            for (int y = 1; y < ROOM_MAX_SIZE - 1; y++)
            {
                for (int x = 1; x < ROOM_MAX_SIZE - 1; x++)
                {
                    int walls = countAdjacentWalls(x, y, room);
                    if (walls > NEIGHBORS)
                    {
                        room[x][y] = 1;
                    }
                    else if (walls < NEIGHBORS)
                    {
                        room[x][y] = 0;
                    }
                }
            }
        }

        // Check if room is valid
        auto [width, height] = getRoomDimensions(room);
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (room[x][y] == 0)
                {
                    return room;
                }
            }
        }
    }
}

std::vector<std::vector<int>> RoomAdditionGenerator::generateCavernRoom()
{
    // Similar to generateCellularRoom but with CAVERN_MAX_SIZE
    // Implementation similar to cellular room but larger
    return generateCellularRoom(); // Simplified for this example
}

void RoomAdditionGenerator::addRoom(int roomX, int roomY, const std::vector<std::vector<int>> &room)
{
    auto [width, height] = getRoomDimensions(room);
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            if (room[x][y] == 0)
            {
                map_[roomX + x][roomY + y] = 0;
            }
        }
    }
    rooms_.push_back(room);
}

void RoomAdditionGenerator::addTunnel(std::pair<int, int> wallTile, std::pair<int, int> direction, int tunnelLength)
{
    int startX = wallTile.first + direction.first * tunnelLength;
    int startY = wallTile.second + direction.second * tunnelLength;

    for (int i = 0; i < MAX_TUNNEL_LENGTH; i++)
    {
        int x = startX - direction.first * i;
        int y = startY - direction.second * i;
        map_[x][y] = 0;

        if (x + direction.first == wallTile.first &&
            y + direction.second == wallTile.second)
        {
            break;
        }
    }
}

std::pair<int, int> RoomAdditionGenerator::getDirection()
{
    const std::vector<std::pair<int, int>> directions = {
        {0, -1}, {0, 1}, {1, 0}, {-1, 0} // North, South, East, West
    };
    return directions[std::uniform_int_distribution<>(0, 3)(gen)];
}

std::tuple<int, int, std::pair<int, int>, std::pair<int, int>, int> RoomAdditionGenerator::placeRoom(
    const std::vector<std::vector<int>> &room)
{

    auto [roomWidth, roomHeight] = getRoomDimensions(room);

    for (int attempt = 0; attempt < PLACE_ROOM_ATTEMPTS; attempt++)
    {
        auto direction = getDirection();

        // Find valid wall tile
        std::pair<int, int> wallTile;
        bool found = false;

        for (int tries = 0; tries < 100 && !found; tries++)
        {
            int x = std::uniform_int_distribution<>(1, width_ - 2)(gen);
            int y = std::uniform_int_distribution<>(1, height_ - 2)(gen);

            if (map_[x][y] == 1 &&
                map_[x + direction.first][y + direction.second] == 1 &&
                map_[x - direction.first][y - direction.second] == 0)
            {
                wallTile = {x, y};
                found = true;
            }
        }

        if (!found)
            continue;

        // Try to place room
        for (int tunnelLength = 0; tunnelLength < MAX_TUNNEL_LENGTH; tunnelLength++)
        {
            int roomX = wallTile.first - direction.first * tunnelLength;
            int roomY = wallTile.second - direction.second * tunnelLength;

            if (canPlaceRoom(room, roomX, roomY))
            {
                return {roomX, roomY, wallTile, direction, tunnelLength};
            }
        }
    }

    return {-1, -1, {-1, -1}, {-1, -1}, -1};
}

bool RoomAdditionGenerator::canPlaceRoom(const std::vector<std::vector<int>> &room, int roomX, int roomY)
{
    auto [width, height] = getRoomDimensions(room);

    // Check bounds
    if (roomX < 1 || roomX + width >= width_ - 1 ||
        roomY < 1 || roomY + height >= height_ - 1)
    {
        return false;
    }

    // Check overlap
    for (int x = -1; x <= width; x++)
    {
        for (int y = -1; y <= height; y++)
        {
            if (map_[roomX + x][roomY + y] == 0)
            {
                return false;
            }
        }
    }

    return true;
}

std::pair<int, int> RoomAdditionGenerator::getRoomDimensions(const std::vector<std::vector<int>> &room)
{
    if (room.empty())
        return {0, 0};
    return {static_cast<int>(room.size()), static_cast<int>(room[0].size())};
}

int RoomAdditionGenerator::countAdjacentWalls(int x, int y, const std::vector<std::vector<int>> &room)
{
    int count = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue;
            count += room[x + i][y + j];
        }
    }
    return count;
}

void CityWallsGenerator::generate()
{
    leafs_.reserve(512);
    clearMap();
    leafs_.clear();
    rooms_.clear();

    // Start with all floors
    for (auto &row : map_)
    {
        std::fill(row.begin(), row.end(), 0);
    }

    // Create BSP tree
    auto rootLeaf = std::make_unique<Leaf>(0, 0, width_, height_);
    leafs_.push_back(rootLeaf.get());

    bool splitSuccessfully = true;
    while (splitSuccessfully)
    {
        splitSuccessfully = false;
        for (auto leaf : leafs_)
        {
            if (!leaf->child1 && !leaf->child2)
            {
                if (leaf->w > MAX_LEAF_SIZE ||
                    leaf->h > MAX_LEAF_SIZE ||
                    std::uniform_real_distribution<>(0, 1)(gen) > 0.8)
                {
                    if (leaf->split())
                    {
                        leafs_.push_back(leaf->child1.get());
                        leafs_.push_back(leaf->child2.get());
                        splitSuccessfully = true;
                    }
                }
            }
        }
    }

    createRooms();
    createDoors();
}

void CityWallsGenerator::createRooms()
{
    for (auto &leaf : leafs_)
    {
        if (!leaf->child1 && !leaf->child2)
        {
            int w = std::uniform_int_distribution<>(
                ROOM_MIN_SIZE,
                std::min(ROOM_MAX_SIZE, leaf->w - 1))(gen);
            int h = std::uniform_int_distribution<>(
                ROOM_MIN_SIZE,
                std::min(ROOM_MAX_SIZE, leaf->h - 1))(gen);
            int x = std::uniform_int_distribution<>(
                leaf->x, leaf->x + (leaf->w - 1) - w)(gen);
            int y = std::uniform_int_distribution<>(
                leaf->y, leaf->y + (leaf->h - 1) - h)(gen);

            createRoom(Room(x, y, w, h));
            rooms_.push_back(Room(x, y, w, h));
        }
    }
}

void CityWallsGenerator::createRoom(const Room &room)
{
    // Build Walls
    for (int x = room.x1 + 1; x < room.x2; x++)
    {
        for (int y = room.y1 + 1; y < room.y2; y++)
        {
            map_[x][y] = 1;
        }
    }

    // Build Interior
    for (int x = room.x1 + 2; x < room.x2 - 1; x++)
    {
        for (int y = room.y1 + 2; y < room.y2 - 1; y++)
        {
            map_[x][y] = 0;
        }
    }
}

void CityWallsGenerator::createDoors()
{
    for (auto &room : rooms_)
    {
        auto [centerX, centerY] = room.center();

        enum class Wall
        {
            North,
            South,
            East,
            West
        };
        Wall wall = static_cast<Wall>(
            std::uniform_int_distribution<>(0, 3)(gen));

        int doorX, doorY;
        switch (wall)
        {
        case Wall::North:
            doorX = centerX;
            doorY = room.y1 + 1;
            break;
        case Wall::South:
            doorX = centerX;
            doorY = room.y2 - 1;
            break;
        case Wall::East:
            doorX = room.x2 - 1;
            doorY = centerY;
            break;
        case Wall::West:
            doorX = room.x1 + 1;
            doorY = centerY;
            break;
        }

        map_[doorX][doorY] = 0;
    }
}

void MazeWithRoomsGenerator::generate()
{
    clearMap();

    // Ensure odd dimensions
    int adjustedWidth = (width_ % 2 == 0) ? width_ - 1 : width_;
    int adjustedHeight = (height_ % 2 == 0) ? height_ - 1 : height_;

    // Initialize regions
    regions_ = std::vector<std::vector<int>>(
        adjustedWidth, std::vector<int>(adjustedHeight, -1));
    currentRegion_ = -1;

    // Add rooms
    addRooms(adjustedWidth, adjustedHeight);

    // Generate maze in remaining space
    for (int y = 1; y < adjustedHeight; y += 2)
    {
        for (int x = 1; x < adjustedWidth; x += 2)
        {
            if (map_[x][y] != 1)
                continue;
            growMaze({x, y}, adjustedWidth, adjustedHeight);
        }
    }

    // Connect regions
    connectRegions(adjustedWidth, adjustedHeight);

    // Remove dead ends if needed
    if (!ALLOW_DEAD_ENDS)
    {
        removeDeadEnds(adjustedWidth, adjustedHeight);
    }
}

void MazeWithRoomsGenerator::addRooms(int mapWidth, int mapHeight)
{
    std::vector<Room> rooms;

    for (int i = 0; i < BUILD_ROOM_ATTEMPTS; i++)
    {
        // Generate room with odd dimensions
        int width = (std::uniform_int_distribution<>(
                         ROOM_MIN_SIZE / 2, ROOM_MAX_SIZE / 2)(gen) *
                     2) +
                    1;
        int height = (std::uniform_int_distribution<>(
                          ROOM_MIN_SIZE / 2, ROOM_MAX_SIZE / 2)(gen) *
                      2) +
                     1;

        int x = ((std::uniform_int_distribution<>(
                     0, mapWidth - width - 1)(gen)) /
                 2) *
                    2 +
                1;
        int y = ((std::uniform_int_distribution<>(
                     0, mapHeight - height - 1)(gen)) /
                 2) *
                    2 +
                1;

        Room newRoom(x, y, width, height);

        bool failed = false;
        for (const auto &room : rooms)
        {
            if (newRoom.intersect(room))
            {
                failed = true;
                break;
            }
        }

        if (!failed)
        {
            rooms.push_back(newRoom);
            startNewRegion();
            createRoom(newRoom);
        }
    }
}

void MazeWithRoomsGenerator::growMaze(std::pair<int, int> start, int mapWidth, int mapHeight)
{
    std::vector<std::pair<int, int>> cells;
    std::pair<int, int> lastDir = {0, 0};

    startNewRegion();
    carve(start.first, start.second);
    cells.push_back(start);

    while (!cells.empty())
    {
        auto cell = cells.back();

        std::vector<std::pair<int, int>> unmadeCells;
        const std::vector<std::pair<int, int>> directions = {
            {0, -1}, {0, 1}, {1, 0}, {-1, 0}};

        for (const auto &dir : directions)
        {
            if (canCarve(cell, dir, mapWidth, mapHeight))
            {
                unmadeCells.push_back(dir);
            }
        }

        if (!unmadeCells.empty())
        {
            std::pair<int, int> dir;
            if (std::find(unmadeCells.begin(), unmadeCells.end(), lastDir) != unmadeCells.end() && std::uniform_real_distribution<>(0, 1)(gen) > WINDING_PERCENT)
            {
                dir = lastDir;
            }
            else
            {
                dir = unmadeCells[std::uniform_int_distribution<>(
                    0, unmadeCells.size() - 1)(gen)];
            }

            auto newCell = std::make_pair(
                cell.first + dir.first, cell.second + dir.second);
            carve(newCell.first, newCell.second);

            newCell = std::make_pair(
                cell.first + dir.first * 2, cell.second + dir.second * 2);
            carve(newCell.first, newCell.second);
            cells.push_back(newCell);

            lastDir = dir;
        }
        else
        {
            cells.pop_back();
            lastDir = {0, 0};
        }
    }
}

bool MazeWithRoomsGenerator::canCarve(std::pair<int, int> pos, std::pair<int, int> dir,
                                      int mapWidth, int mapHeight)
{
    int x = pos.first + dir.first * 3;
    int y = pos.second + dir.second * 3;

    if (x <= 0 || x >= mapWidth - 1 || y <= 0 || y >= mapHeight - 1)
    {
        return false;
    }

    x = pos.first + dir.first * 2;
    y = pos.second + dir.second * 2;

    return map_[x][y] == 1;
}

void MazeWithRoomsGenerator::connectRegions(int mapWidth, int mapHeight)
{
    std::vector<std::vector<std::set<int>>> connectorRegions(
        mapWidth, std::vector<std::set<int>>(mapHeight));

    // Find all connectors
    for (int x = 1; x < mapWidth - 1; x++)
    {
        for (int y = 1; y < mapHeight - 1; y++)
        {
            if (map_[x][y] != 1)
                continue;

            std::set<int> regions;
            const std::vector<std::pair<int, int>> directions = {
                {0, -1}, {0, 1}, {1, 0}, {-1, 0}};

            for (const auto &dir : directions)
            {
                int newX = x + dir.first;
                int newY = y + dir.second;
                int region = this->regions_[newX][newY];
                if (region != -1)
                    regions.insert(region);
            }

            if (regions.size() < 2)
                continue;

            connectorRegions[x][y] = regions;
        }
    }

    // Connect regions
    std::map<int, int> merged;
    std::set<int> openRegions;
    for (int i = 0; i <= currentRegion_; i++)
    {
        merged[i] = i;
        openRegions.insert(i);
    }

    while (openRegions.size() > 1)
    {
        // Find a connector
        int connX = -1, connY = -1;
        for (int x = 0; x < mapWidth; x++)
        {
            for (int y = 0; y < mapHeight; y++)
            {
                if (!connectorRegions[x][y].empty())
                {
                    connX = x;
                    connY = y;
                    break;
                }
            }
            if (connX != -1)
                break;
        }

        if (connX == -1)
            break;

        // Merge regions
        map_[connX][connY] = 0;

        std::vector<int> regionsList(
            connectorRegions[connX][connY].begin(),
            connectorRegions[connX][connY].end());

        int dest = regionsList[0];
        std::vector<int> sources(regionsList.begin() + 1, regionsList.end());

        for (int i = 0; i <= currentRegion_; i++)
        {
            if (std::find(sources.begin(), sources.end(), merged[i]) != sources.end())
            {
                merged[i] = dest;
            }
        }

        for (int source : sources)
        {
            openRegions.erase(source);
        }

        // Remove invalid connectors
        for (int x = 0; x < mapWidth; x++)
        {
            for (int y = 0; y < mapHeight; y++)
            {
                if (connectorRegions[x][y].empty())
                    continue;

                std::set<int> newRegions;
                for (int region : connectorRegions[x][y])
                {
                    newRegions.insert(merged[region]);
                }

                if (newRegions.size() == 1)
                {
                    if (std::uniform_real_distribution<>(0, 1)(gen) < CONNECTION_CHANCE)
                    {
                        map_[x][y] = 0;
                    }
                    connectorRegions[x][y].clear();
                }
                else
                {
                    connectorRegions[x][y] = newRegions;
                }
            }
        }
    }
}

void MazeWithRoomsGenerator::removeDeadEnds(int mapWidth, int mapHeight)
{
    bool done = false;
    while (!done)
    {
        done = true;
        for (int y = 1; y < mapHeight - 1; y++)
        {
            for (int x = 1; x < mapWidth - 1; x++)
            {
                if (map_[x][y] == 0)
                {
                    int exits = 0;
                    const std::vector<std::pair<int, int>> directions = {
                        {0, -1}, {0, 1}, {1, 0}, {-1, 0}};

                    for (const auto &dir : directions)
                    {
                        if (map_[x + dir.first][y + dir.second] == 0)
                        {
                            exits++;
                        }
                    }

                    if (exits <= 1)
                    {
                        map_[x][y] = 1;
                        done = false;
                    }
                }
            }
        }
    }
}

void MazeWithRoomsGenerator::startNewRegion()
{
    currentRegion_++;
}

void MazeWithRoomsGenerator::carve(int x, int y)
{
    map_[x][y] = 0;
    regions_[x][y] = currentRegion_;
}

void MazeWithRoomsGenerator::createRoom(const Room &room)
{
    for (int x = room.x1; x < room.x2; x++)
    {
        for (int y = room.y1; y < room.y2; y++)
        {
            carve(x, y);
        }
    }
}

void MessyBSPGenerator::generate()
{
    leafs_.reserve(512);
    clearMap();
    leafs_.clear();

    // Create root leaf
    auto rootLeaf = std::make_unique<Leaf>(0, 0, width_, height_);
    leafs_.push_back(rootLeaf.get());

    // Split leafs
    bool splitSuccessfully = true;
    while (splitSuccessfully)
    {
        splitSuccessfully = false;

        for (auto leaf : leafs_)
        {
            if (!leaf->child1 && !leaf->child2)
            {
                if (leaf->w > MAX_LEAF_SIZE ||
                    leaf->h > MAX_LEAF_SIZE ||
                    std::uniform_real_distribution<>(0, 1)(gen) > 0.8)
                {
                    if (leaf->split())
                    {
                        leafs_.push_back(leaf->child1.get());
                        leafs_.push_back(leaf->child2.get());
                        splitSuccessfully = true;
                    }
                }
            }
        }
    }

    // Create rooms
    for (auto &leaf : leafs_)
    {
        if (!leaf->child1 && !leaf->child2)
        {
            createRooms(*leaf);
        }
    }

    // Connect rooms
    for (size_t i = 0; i < leafs_.size() - 1; i++)
    {
        if (leafs_[i]->room && leafs_[i + 1]->room)
        {
            connectRooms(*leafs_[i]->room, *leafs_[i + 1]->room);
        }
    }

    // Clean up
    if (SMOOTH_EDGES)
    {
        smoothMap();
    }
}

void MessyBSPGenerator::createRooms(Leaf &leaf)
{
    int w = std::uniform_int_distribution<>(
        ROOM_MIN_SIZE,
        std::min(ROOM_MAX_SIZE, leaf.w - 1))(gen);
    int h = std::uniform_int_distribution<>(
        ROOM_MIN_SIZE,
        std::min(ROOM_MAX_SIZE, leaf.h - 1))(gen);
    int x = std::uniform_int_distribution<>(
        leaf.x, leaf.x + (leaf.w - 1) - w)(gen);
    int y = std::uniform_int_distribution<>(
        leaf.y, leaf.y + (leaf.h - 1) - h)(gen);

    leaf.room = std::make_unique<Room>(x, y, w, h);
    createRoom(*leaf.room);
}

void MessyBSPGenerator::createRoom(const Room &room)
{
    for (int x = room.x1 + 1; x < room.x2; x++)
    {
        for (int y = room.y1 + 1; y < room.y2; y++)
        {
            if (x > 0 && x < width_ - 1 && y > 0 && y < height_ - 1)
            {
                map_[x][y] = 0;
            }
        }
    }
}

void MessyBSPGenerator::connectRooms(const Room &room1, const Room &room2)
{
    auto [x1, y1] = room1.center();
    auto [x2, y2] = room2.center();

    int currX = x1;
    int currY = y1;

    while (!((room2.x1 <= currX && currX <= room2.x2) &&
             (room2.y1 <= currY && currY <= room2.y2)))
    {
        float north = 1.0f;
        float south = 1.0f;
        float east = 1.0f;
        float west = 1.0f;

        // Weight based on target position
        if (currX < x2)
            east += 1.0f;
        else if (currX > x2)
            west += 1.0f;
        if (currY < y2)
            south += 1.0f;
        else if (currY > y2)
            north += 1.0f;

        // Normalize weights
        float total = north + south + east + west;
        north /= total;
        south /= total;
        east /= total;
        west /= total;

        // Choose direction
        float choice = std::uniform_real_distribution<>(0, 1)(gen);
        int dx = 0, dy = 0;

        if (choice < north)
        {
            dy = -1;
        }
        else if (choice < north + south)
        {
            dy = 1;
        }
        else if (choice < north + south + east)
        {
            dx = 1;
        }
        else
        {
            dx = -1;
        }

        // Move and carve
        if (currX + dx > 0 && currX + dx < width_ - 1 &&
            currY + dy > 0 && currY + dy < height_ - 1)
        {
            currX += dx;
            currY += dy;
            map_[currX][currY] = 0;
        }
    }
}

void MessyBSPGenerator::smoothMap()
{
    for (int i = 0; i < 3; i++)
    {
        for (int x = 1; x < width_ - 1; x++)
        {
            for (int y = 1; y < height_ - 1; y++)
            {
                int walls = countAdjacentWalls(x, y);

                if (map_[x][y] == 1 && walls <= SMOOTHING)
                {
                    map_[x][y] = 0;
                }
                else if (map_[x][y] == 0 && walls >= FILLING)
                {
                    map_[x][y] = 1;
                }
            }
        }
    }
}

int MessyBSPGenerator::countAdjacentWalls(int x, int y)
{
    int count = 0;
    if (map_[x][y - 1] == 1)
        count++;
    if (map_[x][y + 1] == 1)
        count++;
    if (map_[x - 1][y] == 1)
        count++;
    if (map_[x + 1][y] == 1)
        count++;
    return count;
}


} // namespace map_gen


