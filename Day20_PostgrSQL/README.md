# Saurabh Kumar | ESET0128  |   Advance Python AIML DAY:20

### 1. Write SELECT statements with filtering (WHERE)
SELECT * FROM employeedb
WHERE city = 'Surat';
 
### 2. Use ORDER BY and LIMIT clauses
SELECT * FROM employeedb
ORDER BY id DESC
LIMIT 2;

### 3. Insert, update, and delete data
    #### Insert
    INSERT INTO employeedb (id, name, city)
    VALUES (12345, 'Ravi', 'Pune');
    
    #### Update 
    UPDATE employeedb
    SET city = 'Mumbai'
    WHERE name = 'Uday';
    
    #### Delete
    DELETE FROM employeedb
    WHERE id = 11234;
 
### 4. Use DISTINCT and aggregate functions (COUNT, SUM, AVG)
    #### DISTINCT
    SELECT DISTINCT city FROM employeedb;
    
    #### Count
    SELECT COUNT(*) AS total_employees FROM employeedb;
    
 
### 5. Basic string and date/time functions

    #### Convert name to uppercase
    SELECT UPPER(name) AS upper_name FROM employeedb;
    
    #### Show current date and time
    SELECT NOW();



### 1.Create tables with appropriate data types
CREATE TABLE bucket2(id INT PRIMARY KEY,name VARCHAR(50),city VARCHAR(50) DEFAULT 'INDIA',age INT,mobile_num BIGINT NOT NULL UNIQUE);

### 2. Use constraints: NOT NULL, UNIQUE, PRIMARY KEY
### 3. Define default values and check constraints

CREATE TABLE employees (id INT PRIMARY KEY,name VARCHAR(50) NOT NULL,email VARCHAR(100) UNIQUE,department VARCHAR(50) DEFAULT 'General',salary INT CHECK (salary > 0));


### 4. Understand and use SERIAL and UUID types

### 5. Alter tables to add/drop columns and constraints

    #### Add a new column
    ALTER TABLE bucket2
    ADD COLUMN email VARCHAR(100);

    #### Add a constraint

    ALTER TABLE bucket2
    ADD CONSTRAINT chk_age CHECK (age >= 18);

    #### Drop a column

    ALTER TABLE bucket2
    DROP COLUMN city;

    #### Drop a constraint

    ALTER TABLE bucket2
    DROP CONSTRAINT chk_age;



### 1. Implement INNER JOIN, LEFT JOIN, RIGHT JOIN
    #### inner join
    select * from bucket1 AS b INNER JOIN employeedb AS e ON b.id=e.id;

    #### Left Join
    select * from bucket1 AS b LEFT JOIN employeedb AS e ON b.id=e.id;

    #### Right Join
    select * from bucket1 AS b Right JOIN employeedb AS e ON b.id=e.id;	
    select * from bucket1 AS b Right JOIN employeedb AS e ON b.id=e.id;





### 2. Write simple correlated and non-correlated subqueries
    #### correlated subqueries
    select id,name from employeedb where id=(select id from bucket1 where employeedb.id=id);
    select id,name from employeedb where id=(select id from bucket1 where employeedb.name=name);

    select id,name from employeedb where id=(select id from bucket1 where employeedb.id=id);

    #### non-correlated subqueries
    select * from employeedb where employeedb.Age=(select MAX(Age) from bucket1)


### 3. Use EXISTS and IN predicates
    #### EXISTS
    SELECT * FROM employeedb e WHERE EXISTS (SELECT id FROM bucket1 b WHERE e.id = b.id);

    #### IN
    SELECT * FROM employeedb e WHERE e.id IN (SELECT id FROM bucket1);

### 4. UNION and UNION ALL 
    #### UNION

    SELECT id FROM employeedb 
    UNION
    SELECT id FROM bucket1;

    SELECT id,name FROM employeedb
    UNION
    SELECT id,name FROM bucket1;

    #### UNION ALL
    SELECT id FROM employeedb
    UNION ALL
    SELECT id FROM bucket1;

    SELECT id,name FROM employeedb
    UNION ALL
    SELECT id,name FROM bucket1;

