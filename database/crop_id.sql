-- phpMyAdmin SQL Dump
-- version 5.1.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Jun 15, 2021 at 04:35 PM
-- Server version: 10.4.18-MariaDB
-- PHP Version: 7.3.27

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `cropwayydb`
--

-- --------------------------------------------------------

--
-- Table structure for table `crop_id`
--

CREATE TABLE `crop_id` (
  `Sl. No` varchar(6) NOT NULL,
  `Crops` varchar(16) DEFAULT NULL,
  `MSP for Kharif 2020-21` varchar(22) DEFAULT NULL,
  `Increase in MSP (Absolute)` varchar(26) DEFAULT NULL,
  `Return over Cost (in %)` varchar(23) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Dumping data for table `crop_id`
--

INSERT INTO `crop_id` (`Sl. No`, `Crops`, `MSP for Kharif 2020-21`, `Increase in MSP (Absolute)`, `Return over Cost (in %)`) VALUES
('1', 'Paddy ', '1,868', '53', '50'),
('10', 'Urad', '6,000', '300', '64'),
('11', 'Groundnut', '5,275', '185', '50'),
('12', 'Sunflower Seed', '5,885', '235', '50'),
('13', 'Soybean ', '3,880', '170', '50'),
('14', 'Sesamum', '6,855', '370', '50'),
('15', 'Nigerseed', '6,695', '755', '50'),
('16', 'Cotton ', '5,515', '260', '50'),
('2', 'Toria', '4,425', '300', '-'),
('3', 'Jowar (Hybrid)', '2,620', '70', '50'),
('4', 'Jowar (Maldandi)', '2,640', '70', '-'),
('5', 'Bajra', '2,150', '150', '83'),
('6', 'Ragi', '3,295', '145', '50'),
('7', 'Maize', '1,850', '90', '53'),
('8', 'Tur', '6,000', '200', '58'),
('9', 'Moong', '7,196', '146', '50');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `crop_id`
--
ALTER TABLE `crop_id`
  ADD PRIMARY KEY (`Sl. No`),
  ADD UNIQUE KEY `idx` (`Sl. No`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
